from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    try:
        realsense_pkg_dir = get_package_share_directory('realsense2_camera')
        realsense_launch_file = os.path.join(realsense_pkg_dir, 'launch', 'rs_launch.py')
    except Exception as e:
        print(f"Warning: realsense2_camera package not found: {e}")
        realsense_launch_file = None

    # ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
    realsense_camera_launch = []
    if realsense_launch_file:
        realsense_camera_launch.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    'align_depth.enable': 'true', 
                }.items()
            )
        )
    
    detection_manager_node = Node(
        package='computer_vision',
        executable='computer_vision.detection_manager',
        name='detection_manager',
        output='screen'
    )

    detector_node = Node(
        package='computer_vision',
        executable='detector', 
        name='detector',
        output='screen'
    )
    
    set_mode_service_call = ExecuteProcess(
        cmd=['sleep', '5', '&&',    # wait for nodes to be up
             'ros2', 'service', 'call', '/detection/set_mode', 
             'reseq_interfaces/srv/SetMode', '"{mode: 2}"'],
        output='screen',
        shell=True,
    )

    return LaunchDescription(
        realsense_camera_launch + [
            detection_manager_node,
            detector_node,
            set_mode_service_call
        ]
    )