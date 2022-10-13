#!/usr/bin/env python
import rospy
import matplotlib
from openai_ros.msg import RLExperimentInfo
import matplotlib.pyplot as plt

def callback(data):

    global counter
    if counter % 10 == 0:
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("episodes")
        plt.ylabel("cumulated episode rewards")

        episode = data.episode_number
        reward = data.episode_reward
        plt.plot(episode,reward, color='red', linewidth=2.5)

        plt.draw()
        plt.pause(0.00000000001)
    counter+=1
    # matplotlib.rcParams.update({'font.size': 15})



def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('reward_listener', anonymous=True)


    rospy.Subscriber('/openai/reward', RLExperimentInfo, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    counter = 0
    listener()
