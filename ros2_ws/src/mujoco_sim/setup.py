import os
from glob import glob

from setuptools import setup

package_name = 'mujoco_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chris Cox',
    maintainer_email='chrisman4247@gmail.com',
    description='MuJoCo physics backend for the RAMMP Kinova stack via ros2_control.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'build_scene = mujoco_sim.build_scene:main',
            'mirror_viewer = mujoco_sim.mirror_viewer:main',
        ],
    },
)
