#!/bin/bash

# Define the commands you want to run in each tab
COMMANDS=("roslaunch ur_gazebo ur3_bringup.launch" "rviz" "roslaunch ur3_moveit_config moveit_planning_execution.launch sim:=true")

# Create a new tab for each command
for i in "${COMMANDS[@]}"; do
    gnome-terminal --tab -- bash -c "$i; exec bash"
    sleep 3
done
