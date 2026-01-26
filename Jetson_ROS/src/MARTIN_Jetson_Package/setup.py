import os
from setuptools import setup, find_packages

package_name = 'MARTIN_Jetson_Package'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        # ROS 2 core packages
        'rclpy',
        'std-msgs',
        'sensor-msgs',
        'geometry-msgs',
        'cv-bridge',
        
        # Computer Vision and AI
        'ultralytics==8.3.65',  # YOLOv8
        'opencv-python==4.12.0.88',
        'numpy==2.2.6',
        
        # RealSense depth camera
        'pyrealsense2==2.56.5.9235',
        
        # Serialization and ML
        'onnx==1.19.1',
        'onnxruntime==1.23.2',
        'protobuf==6.33.1',
        
        # Supporting libraries
        'optuna==3.6.1',
        'coloredlogs==15.0.1',
        'flatbuffers==25.9.23',
        'humanfriendly==10.0',
        'ml-dtypes==0.5.3',
        'typing-extensions==4.15.0',
    ],
    author='ullmannb',
    author_email='ullmannb@tamu.edu',
    maintainer='ullmannb',
    maintainer_email='ullmannb@tamu.edu',
    url='https://github.com/augyr19/MARTIN',
    description='MARTIN ROS 2 Humble package for trash detection with SYBIL and RealSense',
    long_description="""
    Mobile Autonomous Remover of Trash In the eNvironment (MARTIN) ROS 2 Humble package.
    
    Integrates:
    - SYBIL: Single-class YOLOv8 Based Identifier of Litter
    - RealSense: Intel RealSense D435i depth camera for 3D positioning
    - ROS 2: Real-time inference and visualization nodes
    
    Key Components:
    - sensors/: RealSense camera interface with post-processing filters
    - models/: SYBIL YOLOv8 model wrapper
    - utils/: Depth operations and coordinate transformations
    - scripts/: ROS 2 nodes (sybil_node, camera_node)
    """,
    license='MIT',
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'april_tag_node = MARTIN_Jetson_Package.scripts.AprilTag_node:main',
            'sybil_node = MARTIN_Jetson_Package.scripts.SYBIL_node:main',
            'camera_node = MARTIN_Jetson_Package.scripts.camera_node:main',
        ],
    },
)
