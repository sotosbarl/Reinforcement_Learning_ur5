#!/usr/bin/env python

'''
    Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Based on many other examples around Internet
    Visit our website at www.theconstruct.ai
'''
import gym
import time
import json
import numpy
import random
import pickle
import time
import qlearn
from gym import wrappers
import gym
import os
# ROS packages required
import rospy
import rospkg
from std_msgs.msg import Float64
from std_msgs.msg import String
from openai_ros.msg import RLExperimentInfo
from qlearn_messages.msg import point, element, matrix

# import our training environment
import ur5_gripper_env


def fill_qLearn_message(qlearn_dict):
    q_learn_message = matrix()
    for q_object, q_reward in qlearn_dict.iteritems():
        q_learn_element = element()
        q_learn_element.qlearn_point.state_tag = q_object[0]
        q_learn_element.qlearn_point.action_number = q_object[1]
        q_learn_element.reward.data = q_reward
        q_learn_message.q_learn_matrix.append(q_learn_element)

    return q_learn_message

if __name__ == '__main__':

    rospy.init_node('gripper_gym', anonymous=True)
    publish_Q_learn = True
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('gripper_training')
    path = pkg_path + '/data.json'
    termination_pub = rospy.Publisher('/termination', Float64, queue_size=1)


    if publish_Q_learn:
        qlearn_pub = rospy.Publisher("/q_learn_matrix", matrix, queue_size=1)
    # Create the Gym environment
    env = gym.make('Ur5GripperShow-v0')
    rospy.loginfo ( "Gym environment done")

    reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('gripper_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)

    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/alpha")
    Epsilon = rospy.get_param("/epsilon")
    Gamma = rospy.get_param("/gamma")
    epsilon_discount = rospy.get_param("/epsilon_discount")
    nepisodes = rospy.get_param("/nepisodes")
    nsteps = rospy.get_param("/nsteps")

    with open(path, 'r') as f:
        q_last_session=pickle.load(f)

    # q_last_session = {}

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon, q =q_last_session )
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.loginfo ("STARTING Episode #"+str(x))

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation, flag = env.reset()
        if flag:
            continue
        state = ''.join(map(str, observation))
        box_coordinates_from_camera = observation[3:5] #will be constant
        # Show on screen the actual situation of the robot
        # env.render()
        Q_matrix = qlearn.get_Q_matrix()
        print(len(Q_matrix))


        a_file = open(path, "w")

        pickle.dump(Q_matrix, a_file)

        a_file.close()
        if x%15==0 and x is not 0:
            print("killing training")
            print('last epsilon was',qlearn.epsilon)
            termination_pub.publish(qlearn.epsilon)
            os.system("rosnode kill "+ "/gym_train")

            # termination_pub.publish('kill')
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            print('step',i)
            if publish_Q_learn:
                # Publish in ROS topics the Qlearn data
                Q_matrix = qlearn.get_Q_matrix()
                qlearn_msg = fill_qLearn_message(Q_matrix)
                qlearn_pub.publish(qlearn_msg)


            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)      #!!!!!!!!!!!!!!!!!here we can pass i (current step as an argument)




            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            qlearn.learn(state, action, reward, nextState)

            if done == "success":
                done = True

            if not(done):
                state = nextState
            else:
                rospy.loginfo ("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = x
        reward_msg.episode_reward = cumulated_reward
        reward_pub.publish(reward_msg)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo ( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))



    rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
