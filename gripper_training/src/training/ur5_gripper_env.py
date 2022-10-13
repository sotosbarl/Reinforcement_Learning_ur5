#!/usr/bin/env python
import sys
import gym
import os
import rospy
import time
import numpy as np
import tf
import time
import random
import moveit_commander
from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose
from std_msgs.msg import Empty as EmptyTopicMsg
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import SetModelState

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from opencv_services.srv import TargetPosition
from controllers_connection import ControllersConnection


max_steps = 12
#register the training environment in the gym as an available one
reg = register(
    id='Ur5GripperShow-v0',
    entry_point='ur5_gripper_env:Ur5GripperEnv',
    max_episode_steps = max_steps,
    )


#moveit_commander.roscpp_initialize(sys.argv)


class Ur5GripperEnv(gym.Env):

    def __init__(self):

        # We assume that a ROS node has already been created
        # before initialising the environment


        self.running_step = rospy.get_param("/running_step")
        #self.max_incl = rospy.get_param("/max_incl")


        # Establishes connection with simulator
        self.gazebo = GazeboConnection()
        self.controllers_object = ControllersConnection(namespace="monoped")

        self.action_space = spaces.Discrete(5) #Forward,Left,Right,Up,Down
        self.reward_range = (-np.inf, np.inf)
        self._seed()

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def _render(self,a,close):
        pass

    # Resets the state of the environment and returns an initial observation.
    def _reset(self):

        # 1st: resets the simulation to initial values
        #self.gazebo.pauseSim()
        # self.gazebo.pauseSim()
        self.gazebo.resetWorld()

        # self.gazebo.resetSim()
        self.gazebo.unpauseSim()


        node1 = "/move_group"
        # node2 ="/opencv_extract_object_positions"
        self.controllers_object.reset_monoped_joint_controllers()
        os.system("rosnode kill "+ node1)
        # os.system("rosnode kill "+ node2)

        #os.system("rosrun "+ node)

        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        self.arm_group = moveit_commander.MoveGroupCommander("right_arm")
        self.hand_group = moveit_commander.MoveGroupCommander("gripper")

        # 2nd: Unpauses simulation





        # 3rd: resets the robot to initial conditions
        # self.check_topic_publishers_connection()
        # self.init_test_pose()

        # self.coordinates = [-0.47355601, -0.17604605,  1.11033873]

        #move to the first position
        pose_target = Pose()

        self.arm_group.set_named_target("test2")
        plan1 = self.arm_group.go(wait=True)
        self.hand_group.set_named_target("open")
        plan2 = self.hand_group.go(wait=True)



        #get box to the initial pose if it fell after reset, and when gripper moved to test box_position
        #pick one of 3 kind of boxes randomly
        box_list = ["red","cup","cup_fallen"]
        self.selected = box_list[random.randint(1, 2)]
        random_box_position_x = random.randint(-2, 2)
        random_box_position_y = random.randint(-2,2)

        rospy.wait_for_service('/gazebo/set_link_state')
        try:
           set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        except rospy.ServiceException :
            print("Service call failed")

        state_msg = ModelState()
        state_msg.model_name = self.selected
        state_msg.pose.position.x = -0.44 + 0.04*random_box_position_x
        state_msg.pose.position.y = -0.25 + 0.04*random_box_position_y
        state_msg.pose.position.z = 1.34493
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1

        self.box_set_position_x = state_msg.pose.position.x
        self.box_set_position_y = state_msg.pose.position.y

        set_model_state(state_msg)
        rospy.sleep(4)
        self.coordinates, self.box_lays_down, height_width_ratio = self.get_box_coordinates()
        print('box pose',self.coordinates)

        for i in self.coordinates:
            #problem with object detection
            if np.isnan(i):
                flag=True
                break
            else:
                flag=False

        if not flag:
    #this is an inital approach to the object based on what we see (heuristic)
    #every episode starts from here
            roll, pitch, yaw = 3.14 , 0.7 , -1.57
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

            pose_target.orientation.x = quaternion[0]
            pose_target.orientation.y = quaternion[1]
            pose_target.orientation.z = quaternion[2]
            pose_target.orientation.w = quaternion[3]

            pose_target.position.x = self.coordinates[0] - 0.03
            pose_target.position.y = self.coordinates[1] + 0.25 #0.3
            pose_target.position.z = self.coordinates[2] + 0.25 #0.12

            self.arm_group.set_pose_target(pose_target)

            # self.plan1 = self.arm_group.plan()
            plan1 = self.arm_group.go(wait=True)
            self.arm_group.clear_pose_targets()
            test_pose = self.arm_group.get_current_pose().pose
            # print("test_pose",test_pose)



            # 4th: takes an observation of the initial condition of the robot
            relative_position,gripper_position = self.take_observation(self.coordinates)
            self.best_dist = self.calculate_dist_between_two_Points(gripper_position, self.coordinates)

            distance = self.find_distance_class(self.best_dist)
            self.ratio_classification = self.classify_height_width_ratio(height_width_ratio)
    # distance, up/right/forward etc , Is box down or false?
            observation = [distance, False, self.box_lays_down]

            # 5th: pauses simulation
            self.gazebo.pauseSim()

            return observation,flag
        else:
            observation = []
            return observation,flag

    def _step(self, action):
        self.gazebo.unpauseSim()


        box_position_x = self.coordinates[0] #box_coordinates_from_camera[0]
        box_position_y = self.coordinates[1] #box_coordinates_from_camera[1]
        box_position_z = self.coordinates[2] #box_coordinates_from_camera[2]

        self.up_x = box_position_x + 0.15
        self.up_y = box_position_y + 0.4
        self.up_z = box_position_z + 0.4
        self.down_x = box_position_x - 0.15
        self.down_y = box_position_y - 0.3
        self.down_z = box_position_z - 0.15

        # orientation_limits_roll_up =3.14
        # orientation_limits_pitch_up =3.14
        # orientation_limits_yaw_up =3.14
        # orientation_limits_roll_up =0
        # orientation_limits_pitch_up =0
        # orientation_limits_yaw_up =0

        pose_target = Pose()
        current_pose = self.arm_group.get_current_pose().pose


        print('action',action)
        #TRANSLATION MOVEMENT
        #move up
        # if action == 0 :
        #     current_pose.position.z +=0.03
        #     self.hand_group.set_named_target("open")

        #move down
        if action == 0 :
            current_pose.position.z -=0.035
            self.hand_group.set_named_target("open")
        #move left
        # elif action == 2 :
        #     current_pose.position.x +=0.035
        #     self.hand_group.set_named_target("open")
        #move right
        # elif action == 3 :
        #     current_pose.position.x -=0.035
        #     self.hand_group.set_named_target("open")
        #move back
        elif action == 1 :
            current_pose.position.y -=0.035
            self.hand_group.set_named_target("open")
        #move Forward
        # elif action == 5 :
        #     current_pose.position.y +=0.035
        #     self.hand_group.set_named_target("open")


        explicit_quat =  [current_pose.orientation.x,current_pose.orientation.y,current_pose.orientation.z,current_pose.orientation.w]
        euler = euler_from_quaternion(explicit_quat)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        #ROTATION MOVEMENT

        #turn up
        # if action == 6 :
        #     roll += 0.2
        #     self.hand_group.set_named_target("open")
        # #turn down
        # elif action == 7 :
        #     roll -= 0.2
        #     self.hand_group.set_named_target("open")
        pitch_before_movement =    pitch
        if action == 2 :
                pitch +=0.2
                current_pose.position.z +=0.035

                self.hand_group.set_named_target("open")
        elif action == 3 :
                pitch -=0.2
                self.hand_group.set_named_target("open")
        print('pitch',pitch)
            #turn back
        # elif action == 8 :
        #         yaw +=0.2
        #         self.hand_group.set_named_target("open")
        #
        #     #turn Forward
        # elif action == 9 :
        #         yaw -=0.2
        #         self.hand_group.set_named_target("open")


        q = quaternion_from_euler(roll,pitch,yaw)
        current_pose.orientation.x = q[0]
        current_pose.orientation.y = q[1]
        current_pose.orientation.z = q[2]
        current_pose.orientation.w = q[3]


        relative_position_foo, gripper_position_before_grasp = self.take_observation(self.coordinates)
        box_location , did_fallfoo, did_movedfoo = self.box_condition()


        dist_before_grasp_attempt = self.calculate_dist_between_two_Points(gripper_position_before_grasp, box_location)
        #don't move at all
        print("gripper_position",gripper_position_before_grasp)

        # if action == 10 :
        #     pass
        #GRIPPER MOVEMENT
        if action == 4 :
            self.hand_group.set_named_target("close")
            self.arm_group.set_named_target("test")

        #
        # elif action == 12 :
        #     self.hand_group.set_named_target("close2")
        #     self.arm_group.set_named_target("test2")
        #
        #
        # elif action == 13 :
        #     self.hand_group.set_named_target("close3")
        #     self.arm_group.set_named_target("test2")
        #
        #
        # elif action == 14 :
        #     self.hand_group.set_named_target("close4")
        #     self.arm_group.set_named_target("test2")





        # plan2 = self.hand_group.plan()

        plan2 = self.hand_group.go(wait=True)
        if action==4:
            rospy.sleep(2)

        else:
            self.arm_group.set_pose_target(current_pose)


        self.hand_group.clear_pose_targets()
        # Then we send the command to the robot and let it go
        # for running_step seconds
        # self.gazebo.unpauseSim()
        # self.arm_group.setPlanningTime(10)
        # self.plan1 = self.arm_group.plan()
        self.plan1 = self.arm_group.go(wait=True)
        print('goal sent to arm')
        # self.arm_group.stop()

        self.arm_group.clear_pose_targets()





        box_location , did_fall, did_moved = self.box_condition()
        relative_position, gripper_position = self.take_observation(self.coordinates)



        # self.gazebo.pauseSim()




        # finally we get an evaluation based on what happened in the sim
        #we got gripper position, we got state of the box (if it fell)
        #we need to have the position of the box (but not from camera-just the real location)




 #here we should check if object fell or not and impose reward/punishment!

#########################################-----------------------$#################################3
 #here we should check if gripper was enabled prematurely and punish it really hard.


 #here we should check also for aborted plan, and punish the action really hard
 #we also check the cartesian distance (if gripper has come closer to the object)
        reward,done = self.process_data(gripper_position, did_fall,did_moved)

        if gripper_position[2]<1.13 and action==0:
            reward-=2
        current_dist = self.calculate_dist_between_two_Points(gripper_position,box_location)
        dist_after_grasp_attempt = current_dist
        print("current_dist",current_dist)
        print("best_dist",self.best_dist)

#if motion is aborted because no plan has been found
        if abs(dist_after_grasp_attempt-dist_before_grasp_attempt)<0.01 and action!=4:
            reward-=5

        if action==2 and self.box_lays_down:
            reward+=5

        # if (current_dist < self.best_dist) and action<2:
        #     reward += 700
        #     self.best_dist = current_dist
        # elif current_dist > self.best_dist and action<2:
        #     reward = -700
        if dist_after_grasp_attempt<dist_before_grasp_attempt and action!=4:
            reward+=3
        else:
            reward-=30

        if gripper_position[1]<box_location[1] and action==2:   #gripper is over the object with certainty
            reward+=4
        # reward-=current_dist*1000
        #reward if gripper moves towards box
        # if (action==1 or action==6) and (self.ratio_classification==2 or self.ratio_classification==3) :
        #     reward+=1200
        #
        # #reward if gripper moves towards box
        # if (action==4 or action==7) and (self.ratio_classification==1 or self.ratio_classification==4):
        #     reward+=1200
        #
        # if (action ==2 or action==3) and abs(dist_after_grasp_attempt - dist_before_grasp_attempt)<0.01:
        #     reward-=1000
        # if action==3 and self.box_lays_down:
        #     reward-=600
        # elif action==3 and not self.box_lays_down:
        #     reward+=600
        #
        # if action==2 and self.box_lays_down:
        #     reward+=600
        # if action==2 and not self.box_lays_down:
        #     reward-=600
        #
        if did_fall is True:
            reward -=3

        # else:
        #     reward +=0
        #
        if did_moved is True:
            reward -=3
        #
        # else:
        #     reward +=0
        #
        # if action==2 and pitch_before_movement>1.1:
        #     reward-=600
        # if action==3 and pitch_before_movement<0.1:
        #     reward-=600
        #
        #     #gripping

        if action==4 and dist_before_grasp_attempt>0.1:
            reward-=4

        if action==4 and dist_after_grasp_attempt>0.3:
            reward-=5
            print("grasp FAILED!!!")
            done= True

        if action==4 and dist_after_grasp_attempt<0.4:
            reward+=45
            print("grasp succeeded!!!")
            reward += (2/dist_after_grasp_attempt)


            done= 'success'
        #
        # if action == 4 and current_dist<0.14:
        #     reward+= 400
        # elif action == 4 and current_dist>0.14:
        #     reward-=1900
        # if action != 4 and current_dist<0.14:
        #     reward -= 600

#don't rotate if you are too close
        # if (action==2 or action==3) and dist_before_grasp_attempt<0.15:
        #     reward-=1000

        # descritize distance:

        distance = self.find_distance_class(current_dist)


        state = [distance, did_fall, self.box_lays_down]

        if action==4:
            rospy.sleep(2)
            self.hand_group.set_named_target("open")
            plan2 = self.hand_group.go(wait=True)




        print("the reward for this step is:", reward)
        return state, reward, done,  {}

    def find_distance_class(self,current_dist):
        if current_dist>0.5:
            distance = "distance1"
        elif current_dist>0.4 and current_dist<0.5:
            distance = "distance2"
        elif current_dist>0.35 and current_dist<0.4:
            distance = "distance3"
        elif current_dist>0.325 and current_dist<0.35:
            distance = "distance5"
        elif current_dist>0.3 and current_dist<0.325:
            distance = "distance6"
        elif current_dist>0.275 and current_dist<0.3:
            distance = "distance7"
        elif current_dist>0.25 and current_dist<0.275:
            distance = "distance8"
        elif current_dist>0.225 and current_dist<0.25:
            distance = "distance9"
        elif current_dist>0.2 and current_dist<0.225:
            distance = "distance10"
        elif current_dist>0.175 and current_dist<0.2:
            distance = "distance11"
        elif current_dist>0.15 and current_dist<0.175:
            distance = "distance12"
        elif current_dist>0.125 and current_dist<0.15:
            distance = "distance13"
        elif current_dist>0.1 and current_dist<0.125:
            distance = "distance14"
        elif current_dist>0.075 and current_dist<0.1:
            distance = "distance15"
        elif current_dist>0.05 and current_dist<0.075:
            distance = "distance16"
        elif current_dist>0.025 and current_dist<0.05:
            distance = "distance17"
        elif current_dist>0 and current_dist<0.025:
            distance = "distance18"

        return distance

    def classify_height_width_ratio(self, height_width_ratio):
        if height_width_ratio>1 and height_width_ratio<2:
            ratio_class = '1'
        elif height_width_ratio<1 and height_width_ratio>0.5:
            ratio_class = '2'
        elif height_width_ratio<0.5:
            ratio_class = '3'
        elif height_width_ratio>2:
            ratio_class = '4'
        return ratio_class


    def take_observation (self,box_coordinates):


        rospy.wait_for_service('/gazebo/get_link_state')
        try:
           get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        except rospy.ServiceException :
            print("Service call failed")

        resp_left = get_link_state("robot::gripper140left_inner_finger","world")
        resp_right = get_link_state("robot::gripper140right_inner_finger","world")

        array1=np.zeros(3)
        array2=np.zeros(3)
        array1[0]=resp_left.link_state.pose.position.x
        array1[1]=resp_left.link_state.pose.position.y
        array1[2]=resp_left.link_state.pose.position.z
        array2[0]=resp_right.link_state.pose.position.x
        array2[1]=resp_right.link_state.pose.position.y
        array2[2]=resp_right.link_state.pose.position.z

        gripper_position = (array1 + array2)/2

        relative_pose = gripper_position - box_coordinates
        if relative_pose[0]>0:
            x_relative = "left"
        else:
            x_relative = "right"

        if relative_pose[1]>0:
            y_relative = "front"
        else:
            y_relative = "back"

        if relative_pose[2]>0:
            z_relative = "up"
        else:
            z_relative = "down"

        relative_position = [x_relative,y_relative,z_relative]
        return relative_position ,gripper_position

    def box_condition(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
           get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        except rospy.ServiceException :
            print("Service call failed")

        state_msg = ModelState()
        state_msg.model_name = self.selected



        resp = get_state(self.selected,"world")
        box_location = np.zeros(3)
        box_location[0] = resp.pose.position.x
        box_location[1] = resp.pose.position.y
        box_location[2] = resp.pose.position.z

        explicit_quat = [resp.pose.orientation.x, resp.pose.orientation.y, resp.pose.orientation.z, resp.pose.orientation.w]
        euler = euler_from_quaternion(explicit_quat)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        if abs(roll)>0.1 or abs(pitch)>0.1 or abs(yaw)>0.1:
            did_fall = True
            print('box has fallen down (or moved)!!!!')
        else:
            did_fall = False
        if  abs(box_location[0]-self.box_set_position_x)>0.03 or abs(box_location[1]-self.box_set_position_y)>0.03:
            did_moved = True
            print('box has moved!!!!')
        else:
            did_moved = False
        return box_location, did_fall , did_moved



    def calculate_dist_between_two_Points(self,a,b):
        # a = np.array((p_init.x ,p_init.y, p_init.z))
        # b = np.array((p_end.x ,p_end.y, p_end.z))

        dist = np.linalg.norm(a-b)

        return dist


    def init_test_pose(self):


        self.arm_group.set_named_target("test2")

        # plan1 = self.arm_group.plan()
        self.plan1 = self.arm_group.go(wait=True)



    # def check_topic_publishers_connection(self):
    #
    #     rate = rospy.Rate(10) # 10hz
    #     while(self.takeoff_pub.get_num_connections() == 0):
    #         rospy.loginfo("No susbribers to Takeoff yet so we wait and try again")
    #         rate.sleep();
    #     rospy.loginfo("Takeoff Publisher Connected")
    #
    #     while(self.vel_pub.get_num_connections() == 0):
    #         rospy.loginfo("No susbribers to Cmd_vel yet so we wait and try again")
    #         rate.sleep();
    #     rospy.loginfo("Cmd_vel Publisher Connected")



    def get_box_coordinates(self):

        # flag = True
        # while flag:
        rospy.wait_for_service('TargetPosition')

        # Create the connection to the service.
        server_call = rospy.ServiceProxy('TargetPosition', TargetPosition)

        # Create an object of the type TriggerRequest.
        # target = TargetPosition()

        # Now send the request through the connection
        data = server_call()
        box_x = data.box_position.x
        box_y = data.box_position.y
        box_z = data.box_position.z
        box_width = data.width
        box_height = data.height
        if box_height>2*box_width:
            box_lays_down = False
        else:
            box_lays_down = True

        coordinates = np.zeros(3)
        coordinates[0]=box_x
        coordinates[1]=box_y
        coordinates[2]=box_z
            # if not(np.isnan(coordinates[0]) or np.isnan(coordinates[1]) or np.isnan(coordinates[2])):
            #     flag = False
        print('box lays down', box_lays_down)
        print(box_height)
        print(box_width)
        height_width_ratio = float(box_height)/float(box_width)
        print("height_width_ratio", height_width_ratio)
        return coordinates,box_lays_down, height_width_ratio


    def process_data(self, gripper_position, did_fall,did_moved):

        done = False
        reward = 0

        x_bad = not(self.down_x < gripper_position[0] < self.up_x)
        y_bad = not(self.down_y < gripper_position[1] < self.up_y)
        z_bad = not(self.down_z < gripper_position[2] < self.up_z)

        if x_bad or y_bad or z_bad or did_fall or did_moved:
            rospy.loginfo ("gripper is too far away or box fell")
            print(gripper_position)
            print(did_fall)

            done = True
            reward = -3


        return reward,done
