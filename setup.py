from setuptools import setup
from glob import glob

package_name = 'buggy'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install the model file
        ('share/' + package_name + '/models', glob('buggy/models/*.pth')),
        # Install launch files
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'torch', 'torchvision', 'opencv-python', 'scipy', 'numpy', 'pillow'],
    zip_safe=True,
    maintainer='ashy',
    maintainer_email='ashy@example.com',
    description='Buggy ROS 2 package with lane detection capabilities',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = buggy.detector:main',  # Entry point for the lane detection script
            'imunode = script.imunode:main',
            'ultrasonic = script.ultrasonic:main',

        ],
    },
)

