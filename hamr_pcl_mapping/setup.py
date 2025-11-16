from setuptools import setup
import os
from glob import glob

package_name = 'hamr_pcl_mapping'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bemi',
    maintainer_email='gbemigao@seas.upenn.edu',
    description='Mapping bringup pipeline: RealSense -> SLAM -> PCD -> grid_map',
    license='TODO: License',
)
