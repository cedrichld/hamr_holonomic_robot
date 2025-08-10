from setuptools import setup

package_name = 'hamr_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    py_modules=['hamr_odom_graph'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cedric',
    maintainer_email='cedrich@seas.upenn.edu',
    description='HAMR controller – C++ and Python nodes',
    license='MIT',
    entry_points={
        'console_scripts': [
            'hamr_odom_graph = hamr_odom_graph:main',
        ],
    },
)
