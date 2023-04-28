# flood-emergency-planning application

### Background
This project can be used to solve the Isle of Wight extreme flooding problem by informing Isle of Wight residents of the fastest escape route to the highest point within a 5km radius in time for walking.

### Preconditions
Make sure to set up the geospatial virtual environment by downloading the python libraries contained within **geospatial.m1.yml**. This is required to make the application function. 
Ensure the **config.json** file is within the Material folder. 

### Usage
User can simply run the application **main.py**, additional functionality can be found within **evacuation_app.py**.  

The program will ask the user to enter their current location in British National Grid format. First the user is required to enter the easting coordinate, then the northing coordinate. After that, the program will calculate the highest point within 5 km of the location he entered and return its coordinates. The program also returns the quickest walking path using Naismith's rule from the user's current location to the highest point and displays it on the map.

If it detects that the user has entered a non-number, it will ask the user to re-enter it.
The user can keep entering new location coordinates and the program will recalculate the highest point within 5 km of the location and re-route it.
If the user wants to end the program, he/she/they can type **quit**.

### Contributors
This project exists thanks to all the people who contribute:

Adam Morgan,<br>
Jia Shi,<br>
Jiayun Kang,<br>
Yiru Xu,<br>
Zhongxian Wang