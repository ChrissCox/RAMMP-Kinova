import os
from glob import glob

from setuptools import setup

package_name = 'curobo_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'voice'), glob('voice/*.html')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chris Cox',
    maintainer_email='chrisman4247@gmail.com',
    description='cuRobo motion planning for the Kinova Gen3 in simulation, with an NL target layer.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner = curobo_planner.planner_node:main',
            'goto = curobo_planner.nl_command:main',
        ],
    },
)
