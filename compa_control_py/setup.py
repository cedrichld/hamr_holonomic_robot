from setuptools import find_packages, setup

package_name = 'compa_control_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cedric',
    maintainer_email='cedrich@seas.upenn.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "compa_pid_graph = compa_control_py.compa_pid_graph:main",
            "compa_odom_graph = compa_control_py.compa_odom_graph:main",
            "compa_controller = compa_control_py.compa_controller:main",
            "tf_trail_node = compa_control_py.tf_trail_node:main",
            "rpy_tf_graph = compa_control_py.rpy_tf_graph:main",
        ],
    },
)
