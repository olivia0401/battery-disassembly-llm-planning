from setuptools import setup
import os
from glob import glob

package_name = 'battery_dismantle_task'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Install all config files, respecting subdirectories
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config/moveit'), glob('config/moveit/*')),
        (os.path.join('share', package_name, 'config/rviz'), glob('config/rviz/*')),
        # Install custom URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        # Install custom gripper macro (fixes isaac_joint_commands compatibility issue)
        (os.path.join('share', package_name, 'grippers/robotiq_2f_85/urdf'),
         glob('grippers/robotiq_2f_85/urdf/*.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Olivia',
    maintainer_email='olivia@example.com',
    description='LLM-controlled battery disassembly task using Kinova Gen3 with Robotiq gripper',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'skill_server_node = battery_dismantle_task.skill_server:main',
            'visual_state_manager_node = battery_dismantle_task.visual_state_manager:main',
            'publish_initial_joint_states_node = battery_dismantle_task.publish_initial_joint_states:main',
            'interactive_joint_control_node = battery_dismantle_task.interactive_joint_control:main',
        ],
    },
)
