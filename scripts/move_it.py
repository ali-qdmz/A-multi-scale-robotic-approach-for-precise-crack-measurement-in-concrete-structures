#!/usr/bin/env python
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
import sys 
import math

def calculate_distance(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)


def move_robot(x, y, z):
    # rospy.init_node('move_ur3_robot', anonymous=True)

    # Initialize the move_group API
    moveit_commander.roscpp_initialize(sys.argv)
    
    # Initialize the move group for the ur3 arm
    arm = moveit_commander.MoveGroupCommander('manipulator')
    
    # Specify target pose
    target_pose = PoseStamped()
    target_pose.header.frame_id = "base_link"
    target_pose.pose.position.x = x
    target_pose.pose.position.y = y
    target_pose.pose.position.z = z
    target_pose.pose.orientation.x = 0.0
    target_pose.pose.orientation.y = 0.0
    target_pose.pose.orientation.z = 0.0
    target_pose.pose.orientation.w = 0.0
    
    max_reach = 0.5  # maximum reach of UR3 robot in meters
    target_distance = calculate_distance(target_pose.pose.position.x, 
                                         target_pose.pose.position.y, 
                                         target_pose.pose.position.z)

    # if target_distance > max_reach:
    #     rospy.loginfo("Target is outside of robot workspace.")
    #     return  # terminate function if target is out of reach

    # Set the target pose
    arm.set_pose_target(target_pose)
    
    # Plan the trajectory
    planned_path = arm.plan()  # unpack the tuple returned by arm.plan()
    # print(planned_path[1])
    # Execute the planned trajectory
    arm.execute(planned_path[1])

# if __name__ == '__main__':
#     try:
#         move_robot(.2, 0, .6)
#     except rospy.ROSInterruptException:
#         pass
