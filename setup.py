from setuptools import find_packages, setup
from glob import glob
package_name = 'computer_vision'

setup(
 name=package_name,
 version='0.0.0',
 packages=[package_name,
           f'{package_name}.detection_manager',
           f'{package_name}.detector_modules',],
 data_files=[
     ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
     ('share/' + package_name, ['package.xml']),
     (f'share/{package_name}/launch', glob('launch/*launch.py')),
     (f'share/{package_name}/detector_modules/models', glob('computer_vision/detector_modules/models/*')),
     (f'share/{package_name}/hazmat_detection/runs/detect/train/weights', ['hazmat_detection/runs/detect/train/weights/best.pt']),
     (f'share/{package_name}/object_detection/runs/detect/train/weights', ['object_detection/runs/detect/train/weights/best.pt']),
   ],
 install_requires=['setuptools'],
 zip_safe=True,
maintainer='Team ISAAC',
maintainer_email='team.isaac@polito.it',
description='computer vision package for ReseQ robot',
license='GNU GPL v3.0',
tests_require=['pytest'],
 entry_points={
     'console_scripts': [
            'detector = computer_vision.detector:main',
            'computer_vision.detection_manager = computer_vision.detection_manager.detection_manager_node:main',
     ],
   },
)