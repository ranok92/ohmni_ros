#!/usr/bin/env python

import rospy
from obstacle_detector.msg import Obstacles
from std_msgs.msg import String
from ohmni_utils import *
from potential_field_controller import PFControllerContWorld as PFControl 
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler
import math
from tb_msgs.msg import robot_state

class PFControllerNode:
    '''
    A class that creates the node during initialization and acts as a controller 
    for the robot.
    '''
    def __init__(self):
        
        self.controller = PFControl(1, np.pi, 
                                    pos_gain=20,
                                    vel_gain=2,
                                    rep_force_mult=1,
                                    rep_f_limit=2,
                                    attr_f_limit=1,
                                    rep_force_dist_limit=4) #intialize the controller

        #subscribe to topic tracked_obstacles
        
        self.obs_subscriber = rospy.Subscriber('/obstacles', 
                                               Obstacles, 
                                               self.control_callback)
        #subscribe to robot state to get agent state

        self.robot_state_subscriber = rospy.Subscriber('/tb_control/robot_state',
                                                        robot_state,
                                                        self.robot_state_callback)
        #publish command for robot

        self.publisher_rate = rospy.Rate(10)
        self.action_publisher = rospy.Publisher('/tb_cmd_vel', Twist, queue_size=0)
        self.visualize_action = rospy.Publisher('/pf_force', PoseStamped, queue_size=0)
        #self.robot_state_subscriber = rospy.Subscriber('/tb_control/robot_state', )

        self.goal_position = np.array([5, 1]) # [x,y] move forward 5 meters
        self.agent_state = None
        print("Initialized controller class")

        '''
        Future work:
        map, record of trajectory, 
        '''     

    def get_vector_from_rpy(self, roll, pitch, yaw):
        '''
        Return a unit vector in euler coordinates corresponding 
        to the rotations provided by the roll, pitch and yaw.
        The value of the input is in radians
        '''

        vec = np.array([1, 0, 0]) #vector in the xy plane along x axis

        rot_matrix_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                   [math.sin(yaw), math.cos(yaw), 0],
                                   [0, 0, 1]])

        rot_matrix_pitch = np.array([[math.cos(pitch), 0,  math.sin(pitch)],
                                     [0, 1, 0],
                                     [-math.sin(pitch), 0, math.cos(pitch)]])

        rot_matrix_roll = np.array([[1, 0, 0],
                                     [0, math.cos(roll), -math.sin(roll)],
                                     [0, math.sin(roll), math.cos(roll)]])

        return np.matmul(rot_matrix_yaw, vec)



    def robot_state_callback(self, robot_state):

        speed = np.linalg.norm(np.array([robot_state.vel_xM, robot_state.vel_yM]))
        #orientation_from_vel = np.array([robot_state.vel_xM, robot_state.vel_yM])/speed

        yaw_from_vel = math.atan2(robot_state.vel_yM, robot_state.vel_xM) * 180/np.pi 

        yaw_from_state = robot_state.yaw * 180/np.pi 
        orientation = self.get_vector_from_rpy(0,0, robot_state.yaw)

        print("Yaw from state: {}, Orientation : {} ".format(yaw_from_state, orientation[0:2]))
        self.agent_state = {"position" : np.array([robot_state.pos_xM, robot_state.pos_yM]),
                            "orientation" : orientation[0:2],
                            "speed" : speed,
                            "id" : None}

        print("distance from goal : ", np.linalg.norm(self.goal_position- 
                                                      self.agent_state["position"]))


    def control_callback(self, tracked_obstacles):
        '''
        The all encompassing function.
        '''

        obstacle_list = self.get_obstacle_list(tracked_obstacles)

        goal_position = self.get_goal_position() 

        state = self.generate_state(obstacle_list, goal_position)

        action, res_force = self.controller.eval_action(state)



        self.visualize_force(res_force)


        #if len(obstacle_list) > 0:
            #print("---------Obstacles---------")

            #for obs in obstacle_list:
            #    print(obs["position"])

            #print("Action :", action)
        print("Total force :", res_force)
        print("Action to the robot :", action*.1)
        x_lin, z_ang = self.conv_to_robot_control(action*.1)
        
        self.send_twist(x_lin, z_ang)



    def get_obstacle_list(self, tracked_obstacles):
        '''
        Read data from the /tracked_obstacles topic and 
        convert that into a list of obstacle dicts aligned
        to the format of the drone environment.
        Converts 

        All the values are in meters
        '''
        nearby_obstacles = []
        count = 0
        #remove the obstacle at (~0.18, 0) as it is the 
        #stem of the robot itself
        stem_location = np.array([0.18, 0.0])

        error = 0.02
        for obs in tracked_obstacles.circles:
            #print("Obs :", obs.center)
            if stem_location[0]+error >= obs.center.x >= stem_location[0]-error and \
               stem_location[1]+error >= obs.center.y >= stem_location[1]-error:
               '''
               This is the stem of the robot
               '''
               #print("Skipped")
               pass
            else:
                obs_position = np.array([obs.center.x, obs.center.y])
                obs_orientation = np.array([obs.velocity.x, obs.velocity.y])

                obs_speed = np.linalg.norm(obs_orientation)

                obs_id = count #add some sort of tracking later
                
                count += 1

                nearby_obstacles.append({"position" : obs_position,
                                         "orientation" : obs_orientation,
                                         "speed" : obs_speed,
                                         "id" : obs_id
                                         }
                                        )
        return nearby_obstacles

    def get_goal_position(self):
        '''
        Locate goal and return the position
        '''
        return self.goal_position


    def conv_to_robot_control(self, action):
        '''
        Given the resulting agent action, convert that x_linear 
        and z_angular
        :param action: 2 dimensional numpy array 
                       action[0] - speed
                       action[1] - change in orientation 
        '''
        return action[0], action[1]


    def visualize_force(self, res_force):

        res_vector = PoseStamped()

        res_vector.header.frame_id = 'laser'
        res_vector.pose.position.x = 0
        res_vector.pose.position.y = 0
        res_vector.pose.position.z = 0

        res_force_unit = res_force/ np.linalg.norm(res_force)

        yaw = math.atan2(res_force[1], res_force[0])

        #print("Yaw :", yaw*180/np.pi)
        quat = list(quaternion_from_euler(0, 0, yaw, axes='sxyz')) # roll pitch yaw

        res_vector.pose.orientation.x = quat[0]
        res_vector.pose.orientation.y = quat[1]
        res_vector.pose.orientation.z = quat[2]
        res_vector.pose.orientation.w = quat[3]

        self.visualize_action.publish(res_vector)
        self.publisher_rate.sleep()


    def send_twist(self, x_linear, z_angular):
        '''
        Publishes robot control command on /tb_cmd_vel topic
        
        :param x_linear : value of linear motion 
        :param z_angular : value of angular motion
        '''
        print("Twist :", x_linear, z_angular)
        if self.action_publisher is None:
            return
        twist = Twist()
        twist.linear.x = x_linear
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = z_angular *10

        # Only send the zero command once so other devices can take control
    
        self.action_publisher.publish(twist)
        self.publisher_rate.sleep()

    def generate_state(self, obstacle_list, goal_position):
        '''
        Generate the state dictionary that aligns to the 
        state dictionary from the environment
        '''
        agent_state = {
                "position": self.agent_state["position"],
                "orientation": self.agent_state["orientation"],
                "speed": self.agent_state["speed"],
                "id": None
                }

        agent_heading_dir = None

        ghost_state = None

        goal_state = goal_position

        start_pos = None

        obstacles = obstacle_list

        return {"agent_state": agent_state,
            "agent_head_dir": agent_heading_dir,
            "ghost_state": ghost_state,
            "goal_state": goal_state,
            "start_pos": start_pos,
            "obstacles": obstacles,
        }
    


if __name__ == '__main__':
    
    
    rospy.init_node('pf_control', anonymous=True)
    pf_control = PFControllerNode()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down controller!")
