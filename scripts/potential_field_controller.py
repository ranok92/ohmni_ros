import numpy as np 
import pdb
import sys
import math


#from featureExtractor.drone_feature_extractor import angle_between, total_angle_between, norm_2d, dot_2d, project


def angle_between(v1, v2):
    v1_conv = v1.astype(np.dtype("float"))
    v2_conv = v2.astype(np.dtype("float"))
    return np.abs(
        np.arctan2(
            np.linalg.det(np.stack((v1_conv, v2_conv))),
            np.dot(v1_conv, v2_conv),
        )
    )



def total_angle_between(v1, v2):
    """
    Calculate total angle between v1 and v2. Resulting angle is in range [-pi, pi].

    :param v1: first vector.
    :type v1: np.array
    :param v2: second vector.
    :type v2: np.array
    :return: angle between v1 and v2, in range [-pi, pi].
    :rtype: float.
    """
    v1_conv = v1.astype(np.dtype("float"))
    v2_conv = v2.astype(np.dtype("float"))
    return np.arctan2(
        np.linalg.det(np.stack((v1_conv, v2_conv))), np.dot(v1_conv, v2_conv),
    )



def dist_2d(v1, v2):
    return math.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)



def norm_2d(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)



def dot_2d(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]



def project(projection_vect, onto_vect):
    """
    Projects the projection_vect onto the onto_vect, returns scalar
    projection amount.

    Args:
        projection_vect (np.array of shape (2,)): Vector that is to be projected.
        onto_vect (np.array of shape (2,)): Vector that is projected onto.
    """
    return dot_2d(projection_vect, onto_vect) / norm_2d(onto_vect)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_rot_matrix(theta):
    '''
    returns the rotation matrix given a theta value
    '''
    return np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def deg_to_rad(val):
    '''
    converts degree to radians
    '''
    return val*(np.pi)/180


def rad_to_deg(val):
    '''
    converts radians to degrees
    '''
    return val*180/np.pi

'''
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    #this function is for [x, y] coordinates,
    #the input vectors v1 and v2 are [row, col]
    v1_corr = np.array([v1[1], -v1[0]])
    v2_corr = np.array([v2[1], -v2[0]])
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
'''


class PotentialFieldController(object):
    '''
    Potential field controller as used in the social navigation repo
    but adapted for state with agent centric obstacles
    '''
    def __init__(self,
                 pos_gain=5, vel_gain=1,
                 rep_force_mult=200,
                 sigma=100,
                 c_limit=5,
                 attr_f_limit=5,
                 rep_f_limit=5,
                 rep_force_dist_limit=60,
                 goal_threshold=0.5,
                 ):

        self.KP = pos_gain
        self.KV = vel_gain
        self.ETA = rep_force_mult
        self.SIGMA = sigma
        self.conic_limit = c_limit
        self.rep_force_dist_limit = rep_force_dist_limit
        self.pfield = None

        '''
        self.action_array = [np.asarray([-1,0]),np.asarray([-1,1]),
                             np.asarray([0,1]),np.asarray([1,1]),
                             np.asarray([1,0]),np.asarray([1,-1]),
                             np.asarray([0,-1]),np.asarray([-1,-1]),
                             np.asarray([0,0])]
        '''

        self.orientation_array = []
        self.speed_array = []
        self.attr_force_lim = attr_f_limit
        self.rep_force_lim = rep_f_limit
        
        self.globalpath = {}
        self.normalize_force = True
        self.eps = np.finfo(float).eps
        self.goal_threshold = goal_threshold

    def calculate_attractive_force_btwpoints(self, agent_state, goal_state):
        '''
        calculates the attractive force the agent is experiencing given the 
        position of the agent and the goal
        returns a 2 dim vector
        Equation 12. Khatib 1986
        '''
        global_coord = agent_state['position']
        #pdb.set_trace()
        goal_vector = goal_state

        if agent_state['orientation'] is None:
            agent_state['orientation'] = np.array([0, 0])

        if np.linalg.norm(global_coord - goal_vector) < self.goal_threshold:

            return 0

        attr_force = self.KP*(global_coord - goal_vector) - self.KV*agent_state['orientation']
        if self.normalize_force:

            mag = np.hypot(attr_force[0] , attr_force[1])/self.attr_force_lim
            return  - attr_force/mag
        else:
            return  - attr_force



    #calculates the repulsive force the agent experiences given its position and 
    #the position of the obstacles
    def calculate_repulsive_force_btwpoints(self, obs):
        '''
        Equation 18 Khatib 1986
        '''
        obs = obs['position']
        #print("Obs position :", obs)

        rho = np.hypot(obs[0], obs[1])+self.eps
        #print ('rho', rho)
        if rho <= self.rep_force_dist_limit:
            force_vector_x = self.ETA*(1.0/rho - 1/self.rep_force_dist_limit)*(1/rho)*(-obs[0])
            force_vector_y = self.ETA*(1.0/rho - 1/self.rep_force_dist_limit)*(1/rho)*(-obs[1])
            force_mag = np.hypot(force_vector_x , force_vector_y)/self.rep_force_lim

            #print("Force in x :{}, Force in y :{}".format(force_vector_x, force_vector_y))
            if self.normalize_force:
                return (force_vector_x/force_mag , force_vector_y/force_mag)
            else:
                return (force_vector_x, force_vector_y)
        else:
            return (0,0)


    #calculates the potential(not being used for now)
    def calculate_positive_potential(self, agent_state, goal_state):
        pforce = None
        dist = np.linalg.norm(agent_state['position']-goal_state)
        if dist < self.conic_limit:

            pforce = 0.5*self.SIGMA*dist**2

        else:

            pforce = (self.conic_limit*self.SIGMA*dist)-(0.5*self.conic_limit**2)

        return pforce

    #calculates the potential (not being used for now)
    def calculate_negative_potential(self, agent_state, obstacle_state):

        pforce = 0

        dist = np.linalg.norm(agent_state['position']-obstacle_state['position'])+self.eps

        if dist <= self.rep_force_dist_limit:

            pforce = 0.5*self.ETA*(1/dist - 1/self.rep_force_dist_limit)**2

        return pforce



class PFControllerDiscrete(PotentialFieldController):

    def __init__(self,
                 speed_div,
                 orient_div,
                 orient_quant,
                 pos_gain=5, vel_gain=1,
                 rep_force_mult=200,
                 sigma=100,
                 c_limit=5,
                 attr_f_limit=5,
                 rep_f_limit=5,
                 rep_force_dist_limit=60,
                 goal_threshold=0.2,
                 ):
        super().__init__(pos_gain=5, vel_gain=1,
                       rep_force_mult=200,
                       sigma=100,
                       c_limit=5,
                       attr_f_limit=5,
                       rep_f_limit=5,
                       rep_force_dist_limit=60,
                       goal_threshold=0.2,)

        self.define_action_array(speed_div, orient_div, orient_quant)
        

    def define_action_array(self, speed_div, orient_div, orient_quant):

        force_lim = self.attr_force_lim
        self.speed_array = [i/speed_div*force_lim for i in range(speed_div)]
        orient_range = int((orient_div-1)/2)
        deg_array = [(deg_to_rad(i*orient_quant)) for i in range(-orient_range, orient_range, 1)]
        unit_v = np.array([-1, 0])
        rot_matrix_list = [get_rot_matrix(deg) for deg in deg_array]
        self.orientation_array = [np.matmul(rot_matrix, unit_v) for rot_matrix in rot_matrix_list]

    
    def select_action_from_force(self, force, state):
        '''
        takes an action based on the action space of the environment
        and the force being currently being applied on the agent
        '''
        #if force is too small, it doesnt move

        #rotate the force based on the current_heading_dir
        force_rel = np.matmul(get_rot_matrix(deg_to_rad(state['agent_head_dir'])), force)
        similarity_index = np.zeros(len(self.orientation_array))
        for i in range(len(self.orientation_array)):

            similarity_index[i] = angle_between(force_rel, self.orientation_array[i])

        #print('Action taken :', np.argmin(similarity_index))
        #print('similarity_index', similarity_index)
        #pdb.set_trace()
        return int(np.argmin(similarity_index)), 3


    def eval_action(self, state):
        '''
        given the current state take the apropriate action
        '''
        total_force = 0
        agent_state = state['agent_state']
        goal_state = state['goal_state']
        obstacle_state_list = state['obstacles']

        #calculate force due to goal
        
        f_goal = self.calculate_attractive_force_btwpoints(agent_state, goal_state)
        total_force += f_goal
        rep_force = np.array([0.0, 0.0])
        for obs in obstacle_state_list:

            rf = self.calculate_repulsive_force_btwpoints(obs)
            rep_force += rf

        #normalize the repulsive force
        rep_force_norm = (rep_force/norm_2d(rep_force))*self.rep_force_lim

        total_force += rep_force_norm
        orientation, speed = self.select_action_from_force(total_force, state)
        '''
        if orientation!=8:
            rel_action = (orientation - state['agent_head_dir'])%8
        else:
            rel_action = true_action
        #pdb.set_trace()
        '''
        return speed*len(self.orientation_array)+orientation




class PFControllerContWorld(PotentialFieldController):

    def __init__(self,
                 max_speed,
                 max_orient,
                 pos_gain=5, vel_gain=1,
                 rep_force_mult=200,
                 sigma=100,
                 c_limit=5,
                 attr_f_limit=5,
                 rep_f_limit=5,
                 rep_force_dist_limit=60,
                 goal_threshold=0.2,):

        super(PFControllerContWorld, self).__init__(pos_gain, vel_gain,
                                                         rep_force_mult,
                                                         sigma,
                                                         c_limit,
                                                         attr_f_limit,
                                                         rep_f_limit,
                                                         rep_force_dist_limit,
                                                         goal_threshold)

        self.max_speed = max_speed
        self.max_orient = max_orient



    def select_action_from_force(self, force, state):
        '''
        given the resulting force acting on the agent and the 
        current state of the environment returns the appropriate 
        action
        The action is a 2 dim numpy array ranging from -1, 1.
        The dimensions correspond to speed and change in orientation
        respectively
        input: 
            force : 2 dim vector
            state : state dictionary of the environment
        '''
        cur_orientation = state['agent_state']['orientation']
        cur_speed = state['agent_state']['speed']

        theta = total_angle_between(cur_orientation, force)
        new_speed = project(cur_orientation, force)


        print("Current orientation :", cur_orientation)
        print("Force direction :", force)

        norm_speed = min(self.max_speed, new_speed)
        print("New speed :", norm_speed)

        #pdb.set_trace()
        norm_theta = theta/np.pi
        return np.asarray([norm_speed, norm_theta])



    def eval_action(self, state):
        '''
        given the current state take the apropriate action
        '''
        total_force = 0
        agent_state = state['agent_state']
        goal_state = state['goal_state']
        obstacle_state_list = state['obstacles']

        #calculate force due to goal
        
        f_goal = self.calculate_attractive_force_btwpoints(agent_state, goal_state)
        total_force += f_goal
        rep_force = np.array([0.0, 0.0])
        for obs in obstacle_state_list:


            rf = self.calculate_repulsive_force_btwpoints(obs)
            #print("For obstacle position :{}, Force: {}".format(obs["position"], rf))
            rep_force += rf

        if norm_2d(rep_force) > self.rep_force_lim:
            rep_force_norm = (rep_force/norm_2d(rep_force))*self.rep_force_lim
        else:
            rep_force_norm = rep_force

        total_force += rep_force_norm
        action_array = self.select_action_from_force(total_force, state)
        #print("Attractive force :", f_goal)
        #print("Repulsive force :", rep_force_norm)
        #print("total force :", total_force)

        '''
        if orientation!=8:
            rel_action = (orientation - state['agent_head_dir'])%8
        else:
            rel_action = true_action
        #pdb.set_trace()
        '''
        return action_array, f_goal
