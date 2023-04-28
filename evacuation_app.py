import geopandas as gpd
import pandas as pd
from cartopy import crs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import networkx as nx
from rasterio.mask import mask
from rasterio.plot import show
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import fiona
import json
import rtree as index


class Evacuee:
    def __init__(self, x, y):
        self.coord = [x, y]
        self.point = Point(x, y)
        self.gpd = gpd.GeoDataFrame([['evacuee', Point(x, y)]], columns=['name', 'geometry'],
                                    crs='EPSG:27700', index=[0])


class FloodAreaExtent:
    def __init__(self, boundary_shape_file):
        try:
            boundary_geojson = fiona.open(boundary_shape_file)
        except Exception as e:
            print(e)
            quit()

        multipolygon_geojson = next(iter(boundary_geojson))
        self.area_boundary = multipolygon_geojson['geometry']['coordinates']  # Extract coords of the multipolygon

    def contains_location(self, point):
        """ Checks if the evacuees' coordinate is inside the flood extent area

        :param point: point: The (x,y) coordinates of the evacuees location shapely.geometry.Point data format
        :return: True or False
        """

        # For each polygon comprising Isle of Wight test if the input of the user location is within each polygon
        for i in self.area_boundary:
            polygon = Polygon(i[0])
            result = polygon.contains(point)
            if result is True:
                return True

        print(f"This coordinate is not inside flood evacuation area: {point}")
        return False


class DigitalElevationModel:
    def __init__(self, digital_terrain_model):
        try:
            self.dtm_data_reader = rasterio.open(digital_terrain_model)
        except Exception as e:
            print(e)
            quit()

    def highest_point_within_buffer(self, point, distance):
        """ Calculates the (x,y) coordinate of the highest point within the buffer area of the evacuees point coordinate

        :param point: The (x,y) coordinates of the evacuees location shapely.geometry.Point data format
        :param distance: The buffer distance in meters, float data format
        :return: The highest point location (x,y) in shapely.geometry.point data format within the buffer area of the
        evacuees' location
        """
        search_raster, buffer_img, buffer_out_transform = self.__clip_point_buffer(point, distance)
        high_point = self.__highest_point_calculator(search_raster)
        return search_raster, buffer_img, buffer_out_transform, high_point

    def elevation_at_point(self, point):
        """Returns the elevation in metres for the point provided"""
        # Based on info from
        # https://gis.stackexchange.com/questions/317391/python-extract-raster-values-at-point-locations
        # https://geopandas.org/en/stable/gallery/geopandas_rasterio_sample.html
        coord_list = [point.coords[0]]
        try:
            v = [x[0] for x in self.dtm_data_reader.sample(coord_list)]
        except Exception as e:
            v = None

        return v[0]

    def __clip_point_buffer(self, point, search_dist):
        """ Returns a raster clipped to the evacuees' location buffer

        :param point: The (x,y) coordinates of the evacuees location shapely.geometry.Point data format
        :param search_dist: The buffer distance in meters, float data format
        :return: A clipped DTM raster in numpy.ndarray data format
        """
        # 5km buffer created for inputted point
        point_buffer = point.buffer(search_dist)
        point_buffer_gdf = gpd.GeoDataFrame({'geometry': point_buffer}, index=[0])
        point_buffer_gdf.crs = 'EPSG:27700'
        point_buffer_json = [json.loads(point_buffer_gdf.to_json())['features'][0]['geometry']]
        # Extract dtm data from buffer
        out_img, out_transform = mask(self.dtm_data_reader, point_buffer_json, True)
        buffer_img, buffer_out_transform = mask(self.dtm_data_reader, point_buffer_json, crop=True, filled=False)
        return out_img, buffer_img, buffer_out_transform

    def __highest_point_calculator(self, search_raster):
        """ Calculates the (x,y) coordinate of the highest point

        :param search_raster: A DTM raster clipped to the buffer area of the evacuees' location
        in numpy.ndarray data format
        :return: The highest point location (x,y) in shapely.geometry.point data format within the buffer area of the
        evacuees' location
        """
        # Calculate highest point
        highest_pixel = np.where(search_raster[0] == np.max(search_raster[0]))  # highest point pixel location
        # transform from image pixel to geographic/projected (x, y) coordinates
        highest_coords_crs = self.dtm_data_reader.xy(highest_pixel[0], highest_pixel[1])
        highest_point = Point(highest_coords_crs[0][0], highest_coords_crs[1][0])
        return highest_point


class IntegratedTransportNetwork:
    def __init__(self, itn_json, dtm, walking_speed_mh=5000, ascent_speed_mh=600):
        try:
            with open(itn_json, 'r') as f:
                self.integrated_transport_network = json.load(f)
        except Exception as e:
            print(e)
            quit()

        self.walking_speed_ms = walking_speed_mh/3600
        self.ascent_speed_ms = ascent_speed_mh/3600
        self.itn_road_nodes = self.integrated_transport_network['roadnodes']
        self.itn_road_links = self.integrated_transport_network['roadlinks']
        self.idx = index.Index()
        self.point_nodes = []
        self.links_gdf = gpd.GeoDataFrame(columns=['link_name', 'start_node', 'end_node', 'length', 'geometry'])
        self.links_gdf.set_crs('EPSG:27700')
        self.node_elevations = {}
        self.network_graph = nx.MultiDiGraph()
        self.__initialize_node_data(dtm)
        self.__initialize_link_data()

    def __initialize_node_data(self, dtm):
        # Create list of roadnode name and easting and northing coordinate
        for node, value in self.itn_road_nodes.items():
            name = node
            for coordinates in value.values():
                point = [name, [coordinates[0], coordinates[1]]]
                self.point_nodes.append(point)

                # get elevation in meters for coordinate and add to node_elevations dictonary
                elevation_m = dtm.elevation_at_point(Point(coordinates[0], coordinates[1]))
                self.node_elevations[name] = elevation_m

                # add node to network graph
                self.network_graph.add_node(name, pos=(coordinates[0], coordinates[1]))

        # Add a spatial index to the roadnodes
        for n, point in enumerate(self.point_nodes):
            self.idx.insert(n, point[1], point[0])

    def __initialize_link_data(self):

        links_list = []

        for name, link in self.itn_road_links.items():
            length_m = link["length"]
            start_node = link["start"]
            end_node = link["end"]
            coords = link["coords"]
            # Convert coords list to Shapely LineString
            # https://shapely.readthedocs.io/en/stable/reference/shapely.LineString.html
            line_vertices = LineString(coords)

            # store link details as new row at end of geodataframe
            # https://www.statology.org/pandas-add-row-to-dataframe/
            new_row = [name, start_node, end_node, length_m, line_vertices]
            links_list.append(new_row)

            # Get elevation for start and end node
            start_elevation_m = self.node_elevations[start_node]
            end_elevation_m = self.node_elevations[end_node]

            # Calculating time taken walking start node to end node factoring in elevation and distance
            elevation_diff_start_end_m = end_elevation_m - start_elevation_m
            elevation_diff_start_end_m = 0 if elevation_diff_start_end_m <= 0 else elevation_diff_start_end_m
            seconds_walking_start_end = self.__calculate_walking_time(distance_m=length_m,
                                                                      ascent_m=elevation_diff_start_end_m)

            # Calculating time taken walking end node to start node factoring in elevation and distance
            elevation_diff_end_start_m = start_elevation_m - end_elevation_m
            elevation_diff_end_start_m = 0 if elevation_diff_end_start_m <= 0 else elevation_diff_end_start_m
            seconds_walking_end_start = self.__calculate_walking_time(distance_m=length_m,
                                                                      ascent_m=elevation_diff_end_start_m)

            # save link in network graph - twice: once for each direction
            self.network_graph.add_edge(start_node, end_node, length=length_m,
                                        walk_time_seconds=seconds_walking_start_end)

            self.network_graph.add_edge(end_node, start_node, length=length_m,
                                        walk_time_seconds=seconds_walking_end_start)

        # Append links data to geodataframe property
        temp_gdf = gpd.GeoDataFrame(links_list, columns=['link_name', 'start_node', 'end_node', 'length', 'geometry'])
        self.links_gdf = pd.concat([self.links_gdf, temp_gdf], ignore_index=True, sort=False)

    def __calculate_walking_time(self, distance_m, ascent_m):
        walking_time_seconds = (ascent_m * self.ascent_speed_ms) + (distance_m * self.walking_speed_ms)
        return walking_time_seconds

    # Find the road nodes nearest to the highest point and the user coordinates
    def nearest_road_node_to_coordinate(self, point):
        """ Returns the name of the road node nearest to a (x,y) coordinate

        :param point: shapely.geometry.Point data
        :return: The name of the nearest road node in string format
        """
        coord = [point.coords.xy[0][0], point.coords.xy[1][0]]

        nearest_node = list(self.idx.nearest(coordinates=coord, num_results=1, objects=False))
        # Retrieve roadnodes names for both node nearest to the high point and node nearest to the user
        nearest_node_name = self.point_nodes[nearest_node[0]][0]
        return nearest_node_name

    def quickest_evacuation_route_calculator(self, start_node, end_node):

        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path.html
        evacuation_route_nodes = nx.dijkstra_path(self.network_graph, source=start_node,
                                                  target=end_node, weight="walk_time_seconds")

        evacuation_route_links_gdf = self.__get_links_for_nodes(evacuation_route_nodes)
        return evacuation_route_links_gdf

    def __get_links_for_nodes(self, node_list):
        """Retrieve links geodataframe for nodes provided"""
        # Query the geodataframe to retrieve the links and their linestring geometries
        # ie any road link whose start and end nodes are both in the list of evacuation nodes
        try:
            links_gdf = self.links_gdf.loc[(self.links_gdf['start_node'].isin(node_list)) & (
                        self.links_gdf['end_node'].isin(node_list)),
                        ['link_name', 'start_node', 'end_node', 'geometry']].drop_duplicates()
        except:
            links_gdf = None

        return links_gdf


class MapPlotter:
    def __init__(self, background_tif_file, elevation_asc):
        self.background = rasterio.open(background_tif_file)
        self.elevation = rasterio.open(elevation_asc)

    def plot_map(self, user_point_gpd, highest_point_gpd, buffer_img, buffer_out_transform, shortest_path_gpd):

        background_array = self.background.read(1)
        elevation_array = self.elevation.read(1)
        palette = np.array([value for key, value in self.background.colormap(1).items()])
        background_image = palette[background_array]
        bounds = self.background.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        display_extent = [user_point_gpd['geometry'].x - 9000, user_point_gpd['geometry'].x + 9000,
                          user_point_gpd['geometry'].y - 6500, user_point_gpd['geometry'].y + 6500]

        fig = plt.figure(figsize=(5.5, 3), dpi=500)
        ax = fig.add_subplot(1, 1, 1, projection=crs.OSGB())
        ax.imshow(background_image, origin='upper', extent=extent, zorder=0)
        ax.set_extent(display_extent, crs=crs.OSGB())

        # plot the user point, nearby highest point and elevation
        user_point_gpd.plot(ax=ax, marker='o', color='black', markersize=12, zorder=2, label='user point')
        highest_point_gpd.plot(ax=ax, marker='o', color='red', markersize=12, zorder=3, label='highest point')
        rasterio.plot.show(buffer_img, transform=buffer_out_transform, ax=ax, extent=extent, alpha=0.5, zorder=1)
        shortest_path_gpd.plot(ax=ax, edgecolor='blue', linewidth=1.5, zorder=4, label='shortest path')

        # plot the color-bar
        cax = fig.add_axes([0.85, 0.12, 0.03, 0.76])
        im = ax.imshow(elevation_array, cmap='viridis')
        fig.colorbar(im, cax=cax, orientation='vertical', alpha=0.5)
        cax.set_xlabel('Elevation(m)')

        # plot the legend
        ax.legend(loc='lower left', prop={'size': 4.5})

        # plot the north arrow
        # Reference: https://blog.csdn.net/qq_44907989/article/details/125584822
        def add_north(axe, labelsize=7, loc_x=0.93, loc_y=0.95, width=0.04, height=0.09, pad=0.14):
            """
            Draw a scale with 'N' text annotation
            :param axe: The coordinate area to be drawn
            :param labelsize: Display the size of the 'N' text
            :param loc_x: Horizontal proportion of the axe
            :param loc_y: Vertical proportion of the axe
            :param width: The width of the north arrow as a proportion of the axe
            :param height: The height of the north arrow as a proportion of the axe
            :param pad: Gap of text in the proportion of axe
            :return: None
            """
            minx, maxx = axe.get_xlim()
            miny, maxy = axe.get_ylim()
            ylen = maxy - miny
            xlen = maxx - minx
            left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
            right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
            top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
            center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
            triangle = mpatches.Polygon([left, top, right, center], color='k')
            axe.text(s='N', x=minx + xlen * loc_x, y=miny + ylen * (loc_y - pad + height), fontsize=labelsize,
                     horizontalalignment='center', verticalalignment='bottom')
            axe.add_patch(triangle)

        # plot the scalebar
        # Reference：https://stackoverflow.com/questions/32333870/
        # how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot/63494503#63494503

        # plot the scalebar
        # Reference：https://stackoverflow.com/questions/32333870/
        # how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot/63494503#63494503
        def add_scalebar(axe, location=(0.88, 0.05), linewidth=3):
            """
            :param: axe is the axes to draw the scalebar on.
            :param: location is center of the scalebar in axis coordinates.
            :param: linewidth is the thickness of the scalebar.
            """
            # Get the extent of the plotted area according to the coordinate reference system
            x0, x1, y0, y1 = axe.get_extent(crs.OSGB())
            # Define the scalebar location according to the coordinate reference system
            sbx = x0 + (x1 - x0) * location[0]
            sby = y0 + (y1 - y0) * location[1]
            # Calculate a scale bar length
            length = (x1 - x0) / 5000  # in km
            ndim = int(np.floor(np.log10(length)))  # number of digits in number
            length = round(length, -ndim)  # round to 1sf

            # Returns numbers starting with the list

            def scale_number(x):
                if str(x)[0] in ['1', '2', '5']:
                    return int(x)
                else:
                    return scale_number(x - 10 ** ndim)

            length = scale_number(length)
            # calculate x coordinate for the end of the scalebar
            bar_xs = [sbx - length * 500, sbx + length * 500]
            # Plot the scalebar
            axe.plot(bar_xs, [sby, sby], transform=crs.OSGB(), color='k', linewidth=linewidth)
            # Plot the scalebar label
            axe.text(sbx, sby, str(length) + ' km', transform=crs.OSGB(),
                     horizontalalignment='center', verticalalignment='bottom')

        add_scalebar(ax)
        add_north(ax)
        ax.set_title('Evacuation Path for Flood Emergency in the Isle of Wight')
        plt.show()


class EvacuationApp:
    def __init__(self, boundary_shp, elevation_asc, itn_json, background_tif_file, distance: 5000):
        print("Initializing evacuation application ...")
        self.boundary = FloodAreaExtent(boundary_shp)
        self.dtm = DigitalElevationModel(elevation_asc)
        self.integrated_transport_network = IntegratedTransportNetwork(itn_json, self.dtm)
        self.map_plotter = MapPlotter(background_tif_file, elevation_asc)
        self.evacuee = None
        self.distance = distance

    def read_evacuee_coordinate(self):
        valid_input = False
        attempts = 0
        # loop until user enters a valid coordinate or quits
        while not valid_input:
            if attempts > 0:
                print("Input coordinate is not valid, please try again")

            attempts += 1
            input_coord = self.__request_user_input()

            valid_input = self.__validate_and_save_user_input(input_coord)

    @staticmethod
    def __request_user_input():
        print("\nInput your British National Grid coordinate (easting, northing) eg 450500, 90800 or 'quit' to finish")
        input_coord = str(input('What is your coordinate?')) or "450500, 90800"  # This returns a string

        if input_coord.lower() == 'quit':
            print("Thank you for using the Evacuation App")
            quit()

        return input_coord

    def __validate_and_save_user_input(self, input_coord):
        delimiters = [", ", " ", ","]

        have_xy_values = False
        for delimiter in delimiters:
            try:
                x, y = input_coord.split(delimiter)
                have_xy_values = True
                break
            except:
                # We do not have x and y coord values
                continue

        if not have_xy_values:
            return False

        # now try to parse x and y as floats
        try:
            x = float(x)  # This converts it to a float
            y = float(y)  # This converts it to a float
        except ValueError:
            # We do not have valid x and y coord values
            return False

        # if we have got here then we are in business
        # set the evacuee property and return True
        print(f"Evacuee coords: ({x}, {y})")
        self.evacuee = Evacuee(x, y)
        return True

    def evacuate(self):
        """Obtain the coordinates for an evacuee, determine route to safe high point and show map of route.
        Request evacuee coordinates in a continuous loop until all people have been evacuated."""
        while 1 == 1:
            self.read_evacuee_coordinate()

            if self.boundary.contains_location(self.evacuee.point):
                # do evacuation
                print("Loading evacuation route ...")

                # return highest point within the evacuation zone
                search_raster, buffer_img, buffer_out_transform, high_point \
                    = self.dtm.highest_point_within_buffer(self.evacuee.point, self.distance)

                high_point_gpd = gpd.GeoDataFrame([['high point', high_point]], columns=['name', 'geometry'],
                                                  crs='EPSG:27700', index=[0])

                # return road node nearest to evacuee coordinate
                start_node = self.integrated_transport_network.nearest_road_node_to_coordinate(self.evacuee.point)

                # return node nearest to highest point in evacuation zone
                end_node = self.integrated_transport_network.nearest_road_node_to_coordinate(high_point)

                evacuation_route_gdf = self.integrated_transport_network.\
                    quickest_evacuation_route_calculator(start_node, end_node)

                print("Loading map ...")
                self.map_plotter.plot_map(self.evacuee.gpd, high_point_gpd, buffer_img, buffer_out_transform,
                                          evacuation_route_gdf)

            else:
                print("Evacuee location is not in the flood area")
