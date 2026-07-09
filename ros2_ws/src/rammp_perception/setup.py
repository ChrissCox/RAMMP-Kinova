from setuptools import setup

package_name = 'rammp_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chris Cox',
    maintainer_email='chrisman4247@gmail.com',
    description='Continuous scene perception for the RAMMP arm.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = rammp_perception.detector_node:main',
            'probe = rammp_perception.probe:main',
        ],
    },
)
