import rospy
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

def move_robot():
    rospy.init_node('move_ur3_robot', anonymous=True)
    
    # Create an action client
    client = actionlib.SimpleActionClient('/eff_joint_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

    # Wait for the server to come up
    rospy.loginfo('Waiting for action server...')
    client.wait_for_server()
    rospy.loginfo('Action server detected!')

    # Set up the trajectory
    trajectory = JointTrajectory()
    trajectory.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    
    point = JointTrajectoryPoint()
    point.positions = [2, -1, 1.5, -1.57, 1.57, 0]
    point.time_from_start = rospy.Duration(1.0)
    
    trajectory.points.append(point)
    
    # Create a goal
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = trajectory
    
    # Send the goal
    client.send_goal(goal)

    # Wait for result
    client.wait_for_result()

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
