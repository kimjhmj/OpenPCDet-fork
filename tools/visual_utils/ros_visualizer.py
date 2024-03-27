#!/usr/bin/env python3
import os
import sys
import time
import math
import cv2
import copy
import argparse

import rospy
from std_msgs.msg import Header, String, Float32, Float32MultiArray
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import Marker, MarkerArray

def visualize_points(points):
    rospy.init_node("ros_visualizer", anonymous=False)

    test_pub = rospy.Publisher(
        "/ros_visualizer_pointcloud", PointCloud2, queue_size=1
    )
    header = Header()
    header.frame_id = "vehicle"
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("intensity", 12, PointField.FLOAT32, 1),
    ]

    pc = point_cloud2.create_cloud(header, fields, points)
    for i in range(10):
        test_pub.publish(pc)
        print(f'pub...({i+1}/10)')
        rospy.sleep(0.05)
    rospy.sleep(0.5)

