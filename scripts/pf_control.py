#!/usr/bin/env python

import rospy
import numpy as np
from obstacle_detector.msg import Obstacles, SegmentObstacle, CircleObstacle
from std_msgs.msg import String
from ohmni_utils import *
from potential_field_controller import PFControllerContWorld as PFControl 
from potential_field_controller import PIDController as PID
from potential_field_controller import (get_rot_matrix, 
                                       norm_2d, 
                                       angle_between,
                                       unit_vector,
                                       total_angle_between)
from geometry_msgs.msg import (Twist, Vector3,
                              PoseStamped, Quaternion,
                              Point)
from tf.transformations import quaternion_from_euler
import math
from tb_msgs.msg import robot_state


def generate_fake_obstacle(agent_state, pos=None, risk_type='high'):
    '''
    Given the state of the agent places an obstacle that should 
    be of a certain risk type. For debugging purposes
    '''

    circ_obs = CircleObstacle()
    r = 2
    if not pos:
        pos = np.random.rand(2) * r #randomly place an obstacle around the
                                    #agent in a radius of r meters

    agent_vel = agent_state['velocity']
    agent_pos = agent_state['position']
    if risk_type=='high':
        obs_orient = (agent_pos - pos)/np.linalg.norm(agent_pos - pos) 
        vel = obs_orient * .1
        circ_obs.center = Point(pos[0], pos[1], 0)
        circ_obs.velocity = Vector3(vel[0], vel[1], 0)
        circ_obs.radius = 0.2
        circ_obs.true_radius = 0.2

    return circ_obs


def get_intersection_point(p1, p2, p3):
    '''
    Returns the point closest to p3 from a line that goes through
    p1 and p2
    Input:
    :param p1,p2,p3: numpy array specifying the coordinates of the points

    Outpur:
    :param closest_pt: position of the closest point
    '''

    dy, dx = p2[1] - p1[1], p2[0] - p1[0]#for numerical stability 
    
    det = dx*dx + dy*dy
    a = (dy*(p3[1]-p1[1])+dx*(p3[0]-p1[0]))/det
        
    return np.array([p1[0]+a*dx, p1[1]+a*dy])



def inbetween(x1,x2,x3):
    '''
    Return true if x1 is between x2 and x3
    '''
    if x3 >= x1 >= x2 or x2 >= x1 >=x3:
        return True

def get_nearest_point_from_segment(p1, p2, p3):
    '''
    Returns the point closest to p3 on a line segment 
    specified by points p1 and p2
    Input:
    :param p1,p2,p3: numpy array specifying the coordinates of the points

    Outpur:
    :param closest_pt: position of the closest point
    '''
    p4 = get_intersection_point(p1, p2, p3)
    
    if inbetween(p4[0], p1[0], p2[0]):
        
        return p4
    
    else:
        
        if np.linalg.norm(p4 - p1) < np.linalg.norm(p4 - p2):
            
            return p1
        else:
            return p2


def on_robot_path(agent, obstacle,
                  path_length=1,
                  path_width=.4):
    '''
    Given the agent and the obstacle returns true if the 
    obstacle is on the path (both front and back) of the
    robot
    Input
    :param agent    : dict containing agent state
    :param obstacle : dict containing obstacle info

    Output
    :param on_path: Boolean value stating if the obstacle is 
                    on path
    '''
    on_path = False

    obs_width = .2 #fixed for now
    dist_to_maintain_y = path_width/2 + obs_width/2
    dist_to_maintain_x = path_length + obs_width/2

    obs_pos = obstacle["position"]
    if abs(obs_pos[0]) <= dist_to_maintain_x and abs(obs_pos[1]) <= dist_to_maintain_y:

        on_path = True 

    return on_path



def calculate_risk(agent, obstacle, 
                   collison_thresh=0.4,
                   ):
    '''
    Given an agent and an obstacle, calculates the risk of 
    collision of the obstacle to that of the agent
    Input
    :param agent   : dict containg agent state
                     {"position", "orientation", "speed", "id"}
    :param obstacle: dict containing obstacle state
    :param threshold_distance: distance from the agent to gauge 
                               risk posed
    Output
    :param risk: integer indicating the level of risk
                0 - No risk 
                1 - Medium risk
                2 - High risk 
    '''

    rel_pos = obstacle["position"]
    obs_vel = obstacle["orientation"] * obstacle["speed"]
    agent_orientation = agent["orientation"]
    agent_vel = agent["velocity"] 

    #convert agent_vel to local coordinates wrt to its orientation
    diff_theta = total_angle_between(agent_orientation, np.array([1,0]))

    rot_matrix = get_rot_matrix(diff_theta)

    agent_velocity_local_coord = np.matmul(rot_matrix, agent_vel)


    print("Agent vel local coord : ", agent_velocity_local_coord)
    rel_vel = agent_velocity_local_coord - obs_vel

    dist_between = norm_2d(rel_pos)
    
    ang = angle_between(rel_vel, rel_pos)

    min_dist_project = dist_between * math.sin(ang)
    #low risk by default
    #print("Agent state :", agent)
    #print("corrected velocity :", agent_velocity_local_coord)
    risk = 0

    if np.linalg.norm(rel_vel) > 0.02:
        if ang < np.pi /2:
            if min_dist_project < collison_thresh:
                #high risk
                risk = 2

            else:
                #med risk
                risk = 1


    if np.linalg.norm(rel_pos) < 0.3:
        risk=2

    return risk



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
                                    rep_force_dist_limit=.75) #intialize the controller

        #subscribe to topic tracked_obstacles
        
        self.obs_subscriber = rospy.Subscriber('/obstacles', 
                                               Obstacles, 
                                               self.control_callback, 
                                               queue_size=1)
        #subscribe to robot state to get agent state

        self.robot_state_subscriber = rospy.Subscriber('/tb_control/robot_state',
                                                        robot_state,
                                                        self.robot_state_callback,
                                                        queue_size=1)
        #publish command for robot

        self.publisher_rate = rospy.Rate(3)
        
        self.action_publisher = rospy.Publisher('/cmd_vel_accel', Twist, queue_size=0)
        
        #for debugging/visualization purpose
        self.visualize_action = rospy.Publisher('/pf_force', PoseStamped, queue_size=0)

        self.visualize_low_risk_obs = rospy.Publisher('/low_risk_obs', Obstacles, queue_size=0)
        self.visualize_med_risk_obs = rospy.Publisher('/med_risk_obs', Obstacles, queue_size=0)
        self.visualize_high_risk_obs= rospy.Publisher('/high_risk_obs', Obstacles, queue_size=0)
        self.visualize_on_path_obs = rospy.Publisher('/high_risk_obs', Obstacles, queue_size=0)

        self.visualize_goal = rospy.Publisher('/goal_pose', Obstacles, queue_size=0)
        #self.robot_state_subscriber = rospy.Subscriber('/tb_control/robot_state', )


        self.PID_linear = PID(0.5, 0, 0)
        self.PID_angular = PID(0.8, 0, 0)

        self.agent_state = None

        self.goal_position = None # [x,y] move forward 5 meters
        self.set_goal = True

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

        raw_vel = np.array([robot_state.vel_xM, robot_state.vel_yM])

        #discard velocities less than 0.1 as reading is noisy

        vel = np.array([0, 0])
        if np.linalg.norm(raw_vel) > 0.05:
            vel = raw_vel

        yaw_from_state = robot_state.yaw * 180/np.pi 
        orientation = self.get_vector_from_rpy(0,0, robot_state.yaw)

        #print("Yaw from state: {}, Orientation : {} ".format(yaw_from_state, orientation[0:2]))
        #velocity is different from orientation as 
        #orientation is the direction in which the robot is facing 
        #and velocity is the direction in which the robot is moving
        self.agent_state = {"position" : np.array([robot_state.pos_xM, robot_state.pos_yM]),
                            "orientation" : unit_vector(orientation[0:2]),
                            "velocity" : vel, 
                            "speed" : speed,
                            "id" : None}

        #print("Agent speed:", speed)    


    def control_callback(self, tracked_obstacles):
        '''
        The all encompassing function.
        '''

        #obstacle_list = self.get_obstacle_list(tracked_obstacles)
        low, med, high = self.get_risk_based_obstacle_list(tracked_obstacles)


        '''
        if len(low) > 0:
            print("Low risk obstacles :\n", low)

        if len(med) > 0:
            print("Med risk obstacles :\n", med)

        if len(high) > 0:
            print("High risk obstacles :\n", high)
        '''

        if self.set_goal:
            self.set_goal_position(np.array([4, 0])
) 

        state = self.generate_state(high, self.goal_position)

        action, res_force = self.controller.eval_action(state)



        self.visualize_force(res_force)

        self.visualize_goal_position()
        #if len(obstacle_list) > 0:
            #print("---------Obstacles---------")

            #for obs in obstacle_list:
            #    print(obs["position"])

            #print("Action :", action)

        x_lin, z_ang = self.conv_to_robot_control(action*.1)
        
        #print("X linear : {} | Z angular : {}".format(x_lin, z_ang))
        if np.linalg.norm(self.goal_position - self.agent_state["position"]) < 0.2:
            print("goal reached!")
        else:
            #self.send_twist(x_lin, z_ang)
            pass



    def convert_segments_to_circular_obs(self, segment_obs):
        '''
        Given a list of obstacles of type SegmentObstacles,
        converts them to circular obstacles positioned at the
        point situated closest to the agent.
        Input:
        :param segment_obs: list of obstacles of type SegmentObstacle

        Output:
        :param circular_obs: list of obstacles of type CircularObstacle
        '''

        agent_position = self.agent_state["position"]
        agent_speed = self.agent_state["speed"]
        obs_list = []
        count = 0
        stem_location = np.array([0.18, 0.0])
        error = 0.05
        for seg_obs in segment_obs:

            #dropping the Z coordinate 
            first_point = np.array([seg_obs.first_point.x, seg_obs.first_point.y])
            last_point = np.array([seg_obs.last_point.x, seg_obs.last_point.y])

            closest_point = get_nearest_point_from_segment(first_point, 
                                                           last_point, 
                                                           agent_position)

            dx = -(first_point[0] - last_point[0])
            dy = (first_point[1] - last_point[1])
            obs_orientation = np.array([dy, -dx])
            
            if stem_location[0]+error >= closest_point[0] >= stem_location[0]-error and \
               stem_location[1]+error >= closest_point[1] >= stem_location[1]-error:
                '''
                This is the stem of the robot
                '''
                #print("Skipped")
                pass
            else:
                obs_list.append({"position" : closest_point,
                             "orientation" : obs_orientation,
                             "speed" : agent_speed,
                             "id" : count
                             })
            count += 1

        return obs_list



    def get_risk_based_obstacle_list(self, tracked_obstacles):
        '''
        Read data from the /tracked_obstacles topic and 
        convert that into a list of obstacle dicts aligned
        to the format of the drone environment.
        Separates the obstacles in 3 lists based on the perveived
        risk

        All the values are in meters
        '''
        nearby_obstacles = self.convert_segments_to_circular_obs(tracked_obstacles.segments)
        #obs_seq = tracked_obstacles.header.seq
        #nearby_obstacles = []
        count = len(nearby_obstacles)
        #remove the obstacle at (~0.18, 0) as it is the 
        #stem of the robot itself
        stem_location = np.array([0.18, 0.0])

        error = 0.05
        #################################
        '''
        fake_obs = generate_fake_obstacle(self.agent_state)
        nearby_obstacles = []
        tracked_obstacles.circles = []
        tracked_obstacles.circles.append(fake_obs)
        '''
        #################################



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
                                         "orientation" : obs_orientation/obs_speed,
                                         "speed" : 0,
                                         "id" : obs_id
                                         }
                                        )

        #categorize the obstacles based on risk

        #for viewing the obstacles
        view_obs_low_risk = Obstacles()
        view_obs_low_risk.header.frame_id = "laser"
        view_obs_med_risk = Obstacles()
        view_obs_med_risk.header.frame_id = "laser"
        view_obs_high_risk = Obstacles()
        view_obs_high_risk.header.frame_id = "laser"


        high_risk_obs = []
        med_risk_obs = []
        low_risk_obs = []


        #view_obs.header.seq = obs_seq
        #print("perceived obs seq :", view_obs.header.seq)
        
        for obs in nearby_obstacles:

            risk = calculate_risk(self.agent_state, obs)

            on_path = on_robot_path(self.agent_state, obs, 
                                    path_width=.5)

            circ_obs = CircleObstacle()
            pos = obs["position"]
            vel = obs["orientation"] * obs["speed"]
            circ_obs.center = Point(pos[0], pos[1], 0)
            circ_obs.velocity = Vector3(vel[0], vel[1], 0)
            circ_obs.radius = 0.2
            circ_obs.true_radius = 0.2


            if risk==0:
                view_obs_low_risk.circles.append(circ_obs)
                low_risk_obs.append(obs)

            if risk==1:
                view_obs_med_risk.circles.append(circ_obs)
                med_risk_obs.append(obs)  

            if on_path:
                view_obs_high_risk.circles.append(circ_obs)
                high_risk_obs.append(obs)              

        
        #view_obs_high_risk.circles.append()

        self.visualize_high_risk_obs.publish(view_obs_high_risk)
        self.visualize_med_risk_obs.publish(view_obs_med_risk)
        self.visualize_low_risk_obs.publish(view_obs_low_risk)



        return low_risk_obs, med_risk_obs, high_risk_obs


    def get_obstacle_list(self, tracked_obstacles):
        '''
        Read data from the /tracked_obstacles topic and 
        convert that into a list of obstacle dicts aligned
        to the format of the drone environment.
        Converts 

        All the values are in meters
        '''
        nearby_obstacles = self.convert_segments_to_circular_obs(tracked_obstacles.segments)
        #obs_seq = tracked_obstacles.header.seq
        #nearby_obstacles = []
        count = len(nearby_obstacles)
        #remove the obstacle at (~0.18, 0) as it is the 
        #stem of the robot itself
        stem_location = np.array([0.18, 0.0])

        error = 0.05
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

        #for visualization of the nearby obstacles as perceived by the robot

        view_obs = Obstacles()
        view_obs.header.frame_id = "laser"
        #view_obs.header.seq = obs_seq
        #print("perceived obs seq :", view_obs.header.seq)
        for obs in nearby_obstacles:
            circ_obs = CircleObstacle()
            pos = obs["position"]
            vel = obs["orientation"] * obs["speed"]
            circ_obs.center = Point(pos[0], pos[1], 0)
            circ_obs.velocity = Vector3(vel[0], vel[1], 0)
            circ_obs.radius = 0.2
            circ_obs.true_radius = 0.2

            view_obs.circles.append(circ_obs)

        self.visualize_converted_obs.publish(view_obs)
        return nearby_obstacles

    def set_goal_position(self, move, coord='relative'):
        '''
        Locate goal and return the position
        '''

        if coord=='relative':
            agent_current_pos = self.agent_state['position']
            agent_orient = self.agent_state["orientation"]

            diff_ang = total_angle_between(np.array([1, 0]), agent_orient)
            rot_matrix = get_rot_matrix(diff_ang)

            move_global = np.matmul(rot_matrix, move)

            self.goal_position = agent_current_pos + move_global
        else:
            self.goal_position = move

        print("Current state :", self.agent_state)
        print("Diff ang :", diff_ang * 180 / np.pi)
        print("Move :", move_global)
        print("Goal position :", self.goal_position)
        self.set_goal = False


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

        #print("Pose vector visual:", res_vector.pose.orientation)
        self.visualize_action.publish(res_vector)
        #self.publisher_rate.sleep()



    def visualize_goal_position(self):


        goal_pose = Obstacles()
        goal_pose.header.frame_id = 'laser'

        goal = CircleObstacle()
        goal.center = Point(self.goal_position[0], 
                                 self.goal_position[1], 
                                 0)
        goal.velocity = Vector3(0, 0, 0)
        goal.radius = 0.4
        goal.true_radius = 0.2

        goal_pose.circles.append(goal)
        self.visualize_goal.publish(goal_pose)

    def send_twist(self, x_linear, z_angular):
        '''
        Publishes robot control command on /tb_cmd_vel topic
        
        :param x_linear : value of linear motion 
        :param z_angular : value of angular motion
        '''
        if self.action_publisher is None:
            return
        twist = Twist()
        twist.linear.x = x_linear * 2
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = z_angular * 10

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
                "velocity":self.agent_state["velocity"],
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
