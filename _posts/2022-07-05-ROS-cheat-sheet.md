---
title:  "ROS cheat sheet"
excerpt: "Some useful commands and shortcuts for ROS"
toc: true
category:
  - cheat sheet
---


![ROS_logo]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/ros-logo.png)




## Some definitions



The **Robot Operating System (ROS)** is a collection of tools and libraries that helps you build complex and robust robot applications. There are various distributions of ROS, for example ROS Hydro, ROS Indigo, ROS Kinetic, ROS Melodic. ROS 2 was released in 2021.


A [Node](http://wiki.ros.org/Nodes) is process that performs computation and that can communicate with other nodes.


[Topics](http://wiki.ros.org/Topics) are buses over which nodes exchange **messages**. Nodes can publish messages to a topic and/or subscribe to a topic to receive messages.

A [Message](http://wiki.ros.org/Messages) is a data structure (supporting integer, floats, boolean, arrays, etc...) that is used by nodes to communicate with each other.


A **Publisher** is a Node that writes information TO a Topic.

A **Subscriber** is a node that writes information FROM a Topic.


A [Service](http://wiki.ros.org/Services) is a client/server system that allows Nodes to communicate data to each other. Its an alternative to Topics.

[Gazebo](https://gazebosim.org/home) is an open-source physics simulator integrated in ROS. It can set up a world and run robotics simulations. 

[Rviz](http://wiki.ros.org/rviz) is a visualization software, that allows you to view Gazebo data (if you are simulating) or real world data (if you are using the real robot).

A [Package](http://wiki.ros.org/Packages) is a directory where a ROS program is stored.

[Catkin](http://wiki.ros.org/catkin) is the official build system of ROS.

## Cheat sheet

You can download a printable ROS cheat sheet [here]({{ site.url }}{{ site.baseurl }}/assets/downloads/ROS_cheat_sheet.pdf) ([Source](http://air.imag.fr/images/f/f7/ROScheatsheet.pdf)).



### ROS Packages


- Launch a ROS package

```bash 
roslaunch <package_name> <launch_file>
```

For example:

```bash 
# PACKAGE
roslaunch turtlebot_teleop keyboard_teleop.launch
roslaunch bb_8_teleop keyboard_teleop.launch

# PUBLISHER
roslaunch publisher_example move.launch
roslaunch publisher_example stop.launch

# SERVICE
roslaunch service_demo service_launch.launch
rosservice call /service_demo "{}"

# ACTION
roslaunch action_demo action_launch.launch
roslaunch action_demo_client client_launch.launch
```

- List ROS packages

```bash 
rospack list
rospack list | grep <package_name>
```

- Refresh the package list

```bash 
rospack profile
```

- List first order dependencies

```bash 
rospack depends <package_name>
```

- List all dependencies

```bash 
rospack depends <package_name>
```

- Create a package

```bash 
roscd
cd ..
cd src/
catkin_create_pkg <pkg_name> <pkg_dependencies>
cd ..
catkin_make
```
For example,

```bash 
catkin_create_pkg my_package rospy
```

- Install a package from a Github repository

```bash 
cd ~/catkin_ws/src

# clone repo
git clone https://github.com/<...>.git

# check for missing dependencies
rosdep update
rosdep check --from-paths . --ignore-src --rosdistro melodic

# install missing dependencies (if needed)
rosdep install --from-paths . --ignore-src --rosdistro melodic -y

# build package
cd ~/catkin_ws/
catkin_make
```

- Create a Python programme inside a ROS package

```bash 
roscd <package_name>
cd src/
vim <python_file_name.py>
chmod +x <python_file_name.py>
roscd <package_name>
mkdir launch
vim launch/<launch_file.launch>
```

- Compile all packages in catkin directory

```bash 
roscd
catkin_make
. ~/catkin_ws/devel/setup.bash      # Source the setup file
source /opt/ros/melodic/setup.bash  # Source environment setup file
echo $ROS_PACKAGE_PATH              # Print the package path
```

- Compile only one package

```bash 
roscd
catkin_make --only-pkg-with-deps <package_name>
```

### ROS nodes

- List all the nodes currently running

```bash 
rosnode list
```

For example,
```bash 
roscore
rosrun turtlesim turtlesim_node # in another terminal
rosnode list                    # in another terminal  
```

- Run a node within a package

```bash 
rosrun <package_name> <node_name>
```

For example:
```bash 
rosrun turtlesim turtlesim_node
rosrun rviz rviz                   # Run Rviz (debugging)
rosrun rqt_plot rqt_plot
rosrun rqt_graph rqt_graph         # Visualise graph (node connections)
```


- Print info about a node

```bash 
rosnode info /<node_name>
```

- Test connectivity to node

```bash 
rosnode ping
```

- List nodes running on a particular machine or list machines

```bash 
rosnode machine
```

- Kill a running node

```bash 
rosnode kill
```

- Purge registration information of unreachable nodes

```bash 
rosnode cleanup
```


### ROS parameters


- List ROS parameters

```bash 
rosparam list
```

- Get parameter value

```bash 
rosparam get <parameter_name>
```

- Assign a value to a parameter

```bash 
rosparam set <parameter_name> <value>
```

- Load parameters from files

```bash 
rosparam load
```

- Dump parameters to file

```bash 
rosparam dump
```

- Delete parameter

```bash 
rosparam delete
```


### ROS Topics



- List active topics (with verbose option)

```bash 
rostopic list
rostopic list -v
```

- Print the output messages of a topic to the screen in real time

```bash 
rostopic echo /<topic_name>
```

- Publish message with specified value to a topic

```bash 
rostopic pub <topic_name> <message_type> <value>
```

- Display bandwidth used by topic

```bash 
rostopic bw
```

- Display delay of topic from timestamp in header

```bash 
rostopic delay
```

- Find topics by type

```bash 
rostopic find
```

- Display publishing rate of topic

```bash 
rostopic hz
```

- Print topic or field type

```bash 
rostopic type <topic_name>
```


### ROS messages


- Show message description (and its structure)

```bash 
rosmsg show <message>
rosmsg info <message>
```

- This is useful to check the structure of the messages

```bash 
roscd std_msgs/msg/
roscd geometry_msgs/msg/
roscd nav_msgs/msg/
```

- List all messages

```bash 
rosmsg list
```

- Display message called md5sum

```bash 
rosmsg md5
```

- List messages in a package

```bash 
rosmsg package
```

- List packages that contain messages

```bash 
rosmsg packages
```


### Gazebo

- start Gazebo

```bash 
gazebo
```

- run Gazebo server

```bash 
gzserver
```

- run Gazebo client

```bash 
gzclient
```

- kill Gazebo server

```bash 
killall gzserver
```


### Other commands

- Start ROS core

```bash 
roscore
```

- List ROS environment variables

```bash 
export | grep ROS
```

- Go to the Catkin workspace

```bash 
roscd
```

- Go to "package_name"

```bash 
roscd <package_name>
```


### Launch file structure

```bash 
<launch>
<!-- My Package launch file -->
<node pkg="<package_name>" type="<python_file_name.py>" name="<node_name>" output="screen">
</node>
</launch>
```

### Example of publisher node that write messages to a ROS topic


```python 
#! /usr/bin/env python

import rospy          # Import the Python library for ROS
from std_msgs.msg import Int32 # Import the Int32 message from the std_msgs package   

rospy.init_node('topic_publisher') # Initiate a Node named 'topic_publisher'
pub = rospy.Publisher('/counter', Int32, queue_size=1) # Create a Publisher object, that will publish on the /counter topic + messages of type Int32

rate = rospy.Rate(2) # Set a publish rate of 2 Hz
count = Int32() # Create a var of type Int32
count.data = 0 # Initialize 'count' variable

while not rospy.is_shutdown(): # Create a loop that will go until someone stops the program execution
    pub.publish(count) # Publish the message within the 'count' variable
    count.data += 1 # Increment 'count' variable
    rate.sleep() # Make sure the publish rate maintains at 2 Hz
```


### Example of subscriber node that read messages to a ROS topic



```python 
#! /usr/bin/env python

import rospy
from std_msgs.msg import Int32

def callback(msg): # Define a function called 'callback' that receives a parameter named 'msg'
    print msg.data # Print the value 'data' inside the 'msg' parameter

rospy.init_node('topic_subscriber') # Initiate a Node called 'topic_subscriber'

# Create a Subscriber object that will listen
# to the /counter topic and will call the
# 'callback' function each time it reads
# something from the topic
sub = rospy.Subscriber('/counter', Int32, callback) 

rospy.spin()
```

