#contains utility functions to let a controller agent 
#communicate with the robot 

from sensor_msgs.msg import LaserScan
import numpy as np

def deg_to_rad(deg):

    return deg * np.pi / 180


def rad_to_deg(rad):

    return rad * 180 / np.pi



def convert_lidar_point_to_env_coord(index, range_val):
    '''
    Given the lidar index and the corresponding range
    from the lidar sensor, maps the position on the 
    environment.
    
    Coordinate system for the environment is similar to 
    that of an image  ->  x
                     |
                     V
                     
                     y
    Returns (x,y)
    '''
    theta = (360+index-90)%360

    return range_val*np.cos(deg_to_rad(theta)), -range_val*np.sin(deg_to_rad(theta))



def read_obstacles(lidar_info):
    '''
    reads lidar information and creates a state 
    that matches the format of a state dictionary
    returned by our environment
    :param: lidar_info 
    :dtype: sensor_msgs/LaserScan
    '''

    lidar_ranges = lidar_info.ranges 
    '''
    Useful info:
        1. list of length 360. 
        2. 180th index points towards the forward direction 
        3. 90th index points to the right of the robot
        4. The range values are in meters.
        5. Obstacle at around 18 cm at approximately
           180th index is the pole of the robot itself.
        6. Lidar has a range of about 12 meters.
    '''

    nearby_obstacles = []
    print("________________________")
    for i in range(len(lidar_info.ranges)):
        #remove everything outside of 4 meters
        if lidar_info.ranges[i] < .3:
            #print("Robot pole distance {}, direction : {}".format(lidar_info.ranges[i],i))
            loc_x, loc_y = convert_lidar_point_to_env_coord(i, lidar_info.ranges[i])
            obs_position = np.array([loc_x, loc_y])
            
            #####default values for now####
            obs_orientation = np.array([0,-1]) #facing north (on the env)
            obs_speed = 0
            obs_id = None
            ################################
            nearby_obstacles.append({"position" : obs_position,
                                     "orientation" : obs_orientation,
                                     "speed" : obs_speed,
                                     "id" : obs_id
                                     }
                                    )
            print("Found obstacle at position :", obs_position)

    #print(nearby_obstacles)
