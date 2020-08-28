#!/usr/bin/env python

import rospy
from darknet_ros_msgs.msg import BoundingBoxes # Yolo darknet specific msg
from std_msgs.msg import Int8 
from sensor_msgs.msg import Image  
import operator 


def prcb(msg):
    sorted_msg = sorted(msg.bounding_boxes, key=operator.attrgetter('probability'),reverse = True)
    ##To display detection in highest probabilty instead of for loop replace i with sorted_msg[0].Class/probability
    for i in sorted_msg:
        print("class_ID:{}".format(i.Class))
        print("probabiltiy:{}".format(i.probability))
        
    print("-------------------------------------------------------------------------------------------")
    

#This callback will publish the topic read from /image_raw and publishes to topic /camera/rgb/image_raw to which darkent ros subscribes.
#There is more elegant way to do this by using argmap in launch , I am including this for ease of understanding
def imgcb(msg):
    pub=rospy.Publisher('/camera/rgb/image_raw',Image,queue_size=100)
    pub.publish(msg)
    
    
    
    
rospy.init_node('darknet_trial')
rospy.Subscriber('/darknet_ros/bounding_boxes',BoundingBoxes,prcb)
rospy.Subscriber('/image_raw',Image,imgcb) #subscribe to rosbag or simulator image topic




rospy.spin()