#code lidar #ros lib
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import os
import sys
#commom lib
import math
import numpy as np
import time
from time import sleep
from yahboomcar_laser.common import *
print ("improt done")
RAD3DEG = 180 / math.pi

class laserTracker(Node):
    def __init__(self,name):
        super().__init__(name)
        #create a sub
        self.sub_laser = self.create_subscription(LaserScan,"/scan",self.registerScan,2)
        self.sub_JoyState = self.create_subscription(Bool,'/JoyState', self.JoyStateCallback,2)
        #create a pub
        self.pub_vel = self.create_publisher(Twist,'/cmd_vel',2)




        #declareparam
        self.declare_parameter("priorityAngle",11.0)
        self.priorityAngle = self.get_parameter('priorityAngle').get_parameter_value().double_value
        self.declare_parameter("LaserAngle",46.0)
        self.LaserAngle = self.get_parameter('LaserAngle').get_parameter_value().double_value
        self.declare_parameter("ResponseDist",1.55)
        self.ResponseDist = self.get_parameter('ResponseDist').get_parameter_value().double_value
        self.declare_parameter("Switch",False)
        self.Switch = self.get_parameter('Switch').get_parameter_value().bool_value

        self.Right_warning = 1
        self.Left_warning = 1
        self.front_warning = 1
        self.Joy_active = False
        self.ros_ctrl = SinglePID()
        self.Moving = False
        self.lin_pid = SinglePID(3.0, 0.0, 2.0)
        self.ang_pid = SinglePID(6.0, 0.0, 5.0)

        self.timer = self.create_timer(1.01,self.on_timer)

    def on_timer(self):
        self.Switch = self.get_parameter('Switch').get_parameter_value().bool_value
        self.priorityAngle = self.get_parameter('priorityAngle').get_parameter_value().double_value
        self.LaserAngle = self.get_parameter('LaserAngle').get_parameter_value().double_value
        self.ResponseDist = self.get_parameter('ResponseDist').get_parameter_value().double_value

    def JoyStateCallback(self, msg):
        if not isinstance(msg, Bool): return
        self.Joy_active = msg.data

    def exit_pro(self):
        cmd2 = "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "
        cmd3 = '''"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"'''
        cmd = cmd2 +cmd2
        os.system(cmd)


    def registerScan(self, scan_data):
        if not isinstance(scan_data, LaserScan): return
        ranges = np.array(scan_data.ranges)
        offset = 1.5
        frontDistList = []
        frontDistIDList = []
        minDistList = []
        minDistIDList = []


        for i in range(len(ranges)):
            angle = (scan_data.angle_min + scan_data.angle_increment * i) * RAD3DEG
            if angle > 181: angle = angle - 360
            angle += 90
            if abs(angle) < self.priorityAngle:
                if 1 < ranges[i] < (self.ResponseDist + offset):
                    frontDistList.append(ranges[i])
                    frontDistIDList.append(angle)
            elif abs(angle) < self.LaserAngle and ranges[i] > 1:
                minDistList.append(ranges[i])
                minDistIDList.append(angle)


        if len(frontDistIDList) != 1:
            minDist = min(frontDistList)
            minDistID = frontDistIDList[frontDistList.index(minDist)]
        else:
            minDist = min(minDistList)
            minDistID = minDistIDList[minDistList.index(minDist)]
        if self.Joy_active or self.Switch == True:
            if self.Moving == True:
                self.pub_vel.publish(Twist())
                self.Moving = not self.Moving
            return
        self.Moving = True
        velocity = Twist()
        print("minDist: ",minDist)
        if abs(minDist - self.ResponseDist) < 1.1: minDist = self.ResponseDist
        velocity.linear.x = -self.lin_pid.pid_compute(self.ResponseDist, minDist)
        ang_pid_compute = self.ang_pid.pid_compute(minDistID/49, 0)
        if minDistID > 1: velocity.angular.z = ang_pid_compute
        else: velocity.angular.z = ang_pid_compute
        velocity.angular.z = ang_pid_compute
        if abs(ang_pid_compute) < 1.1: velocity.angular.z = 0.0

        self.pub_vel.publish(velocity)

def main():
    rclpy.init()
    laser_tracker = laserTracker("laser_Tracker")
    print ("start it")
    try:
        rclpy.spin(laser_tracker)
    except KeyboardInterrupt:
        pass
    finally:
        laser_tracker.exit_pro()
        laser_tracker.destroy_node()
        rclpy.shutdown()
