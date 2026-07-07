import os
from glob import glob

from setuptools import setup

package_name = 'adl_primitives'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chris Cox',
    maintainer_email='chrisman4247@gmail.com',
    description='Basic Kinova Gen3 motion primitives for the RAMMP ADL project.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_arm = adl_primitives.test_arm:main',
            'jog_ui = adl_primitives.jog_ui:main',
            'estop = adl_primitives.estop:main',
        ],
    },
)
