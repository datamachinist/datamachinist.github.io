---
title:  "Part 8.3 - Getting started with Pybullet"
excerpt: "An introduction to Pybullet, an open-source physics engine simulator for robotics."
toc: true
category:
  - reinforcement learning
  - custom Gym environment
---


[Pybullet](https://pybullet.org/wordpress/) is an open-source physics engine simulator for robotics. We will briefly explain how to use it.

## Simple robot simulation

Please follow the installation instructions [here](https://github.com/bulletphysics/bullet3). Let's check that everything was installed correctly. Run this code, say hi to Pybullet and play with R2D2.

```python
import pybullet as p
import time
import pybullet_data

# Start pybullet simulation
p.connect(p.GUI)    
# p.connect(p.DIRECT) # don't render

# load urdf file path
p.setAdditionalSearchPath(pybullet_data.getDataPath()) 

# load urdf and set gravity
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)

# step through the simluation
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

<iframe width="560" height="315" src="https://www.youtube.com/watch?v=f-mHFv5uelE" title="r2d2 env" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The Pybullet commands are described in the [documentation](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit). The robot's geometry and mechanics are described in XML files and can be loaded in Pybullet. The XML file format compatible with Pybullet are:
- **URDF**: the most common format. Many robots have public URDF file. URDF files are used by the ROS project (Robot Operating System), see [here](http://wiki.ros.org/urdf/Tutorials).
- **SDF**: this file format was developed as part of the Gazebo robot simulator, see [here](http://sdformat.org/).
- **MJCF**: file format developed for the [MuJoCo](https://mujoco.org/) physics engine.

Many of these files are already included in Pybullet, see [here](https://github.com/bulletphysics/bullet3/tree/master/data). Let's see another example where we import a Kuka robot. First, let's download the Kuka urdf files from the ROS Github.

```bash
git clone https://github.com/ros-industrial/kuka_experimental.git
```

Then run this code:

```python
import pybullet as p
import pybullet_data
import time

# start pybullet simulation
p.connect(p.GUI)

# reset the simulation to its original state
p.resetSimulation()

# load urdf file path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# import plane
plane = p.loadURDF("plane.urdf")

# import Kuka urdf and fix it to the ground
robot = p.loadURDF("kuka_experimental/kuka_kr210_support/urdf/kr210l150.urdf", [0, 0, 0], useFixedBase=1)

# request the position and orientation of the robot
position, orientation = p.getBasePositionAndOrientation(robot)
print("The robot position is {}".format(position))
print("The robot orientation (x, y, z, w) is {}".format(orientation))

# print the number of joints of the robot
nb_joints = p.getNumJoints(robot)
print("The robot is made of {} joints.".format(nb_joints))
print("The arm does not really have 8 joints. It has 6 revolute joints and 2 fixed joints.")

# print information about joint 2
joint_index = 2
joint_info = p.getJointInfo(robot, joint_index)
print("Joint index: {}".format(joint_info[0]))
print("Joint name: {}".format(joint_info[1]))
print("Joint type: {}".format(joint_info[2]))
print("First position index: {}".format(joint_info[3]))
print("First velocity index: {}".format(joint_info[4]))
print("flags: {}".format(joint_info[5]))
print("Joint damping value: {}".format(joint_info[6]))
print("Joint friction value: {}".format(joint_info[7]))
print("Joint positional lower limit: {}".format(joint_info[8]))
print("Joint positional upper limit: {}".format(joint_info[9]))
print("Joint max force: {}".format(joint_info[10]))
print("Joint max velocity {}".format(joint_info[11]))
print("Name of link: {}".format(joint_info[12]))
print("Joint axis in local frame: {}".format(joint_info[13]))
print("Joint position in parent frame: {}".format(joint_info[14]))
print("Joint orientation in parent frame: {}".format(joint_info[15]))
print("Parent link index: {}".format(joint_info[16]))

# print state of joint 2
joints_index_list = range(nb_joints)
joints_state_list = p.getJointStates(robot, joints_index_list)

print("Joint position: {}".format(joints_state_list[joint_index][0]))
print("Joint velocity: {}".format(joints_state_list[joint_index][1]))
print("Joint reaction forces (Fx, Fy, Fz, Mx, My, Mz): {}".format(joints_state_list[joint_index][2]))
print("Torque applied to joint: {}".format(joints_state_list[joint_index][3]))

# print state of link 2
link_state_list = p.getLinkState(robot, 2)
print("Link position (center of mass): {}".format(link_state_list[0]))
print("Link orientation (center of mass): {}".format(link_state_list[1]))
print("Local position offset of inertial frame: {}".format(link_state_list[2]))
print("Local orientation offset of inertial frame: {}".format(link_state_list[3]))
print("Link frame position: {}".format(link_state_list[4]))
print("Link frame orientation: {}".format(link_state_list[5]))

# Define gravity in x, y and z
p.setGravity(0, 0, -9.81)

# define a target angle position for each joint (note, you can also control by velocity or torque)
p.setJointMotorControlArray(robot, joints_index_list, p.POSITION_CONTROL, targetPositions=[-1, 0, -0.5, 1, 1, 0, 0, 0])  

joint_index = 2
# step through the simulation
for _ in range(1000000):
    p.stepSimulation()
    time.sleep(1./30.)  # slow down the simulation

    joints_state_list = p.getJointStates(robot, joints_index_list)
    print("Joint position: {}".format(joints_state_list[joint_index][0]))
    print("Joint velocity: {}".format(joints_state_list[joint_index][1]))
    print("Torque applied to joint: {}".format(joints_state_list[joint_index][3]))


p.disconnect()
```


<iframe width="560" height="315" src="https://www.youtube.com/watch?v=W2wM702lsKc&" title="kuka env" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## The Pybullet - Gym interface

Ok so we know how to import robots in Pybullet. Now, how do we train them? We need a way to interact with the simulation using the Gym interface. Fortunately, Pybullet interfaces very nicely with Gym using its pybullet_envs library. For example, you can import the cart-pole environment this way:


![cartpole_pybullet]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/cartpole_pybullet.png)

You can also visualise pre-trained environments [here](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/baselines). For example, see this Kuka grasping robot following a continuous downward policy using the following command.

```bash
python -m pybullet_envs.baselines.enjoy_kuka_diverse_object_grasping
```

However, the environments found in pybullet_envs are not exactly the same as those offered by MuJoCo. Fortunately, the [Pybullet-gym](https://github.com/benelot/pybullet-gym) library has just re-implemented most [MuJoCo](https://github.com/deepmind/mujoco) and [Roboschool](https://openai.com/blog/roboschool/) environments in Pybullet and they seamlessly integrate with OpenAI Gym. For example, the MuJoCo reacher environment can be loaded using [this code](https://github.com/PierreExeter/gym-reacher).


![cartpole_pybullet]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/pybullet-gym.png)