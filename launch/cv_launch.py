import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration('mode')
    skip_rs = LaunchConfiguration('skip_realsense').perform(context)

    # When sensors_launch already starts the RealSense under /realsense/ namespace,
    # remap the detector's camera topics to match
    remappings = []
    if skip_rs == 'true':
        remappings = [
            ('/camera/color/image_raw', '/realsense/color/image_raw'),
            (
                '/camera/aligned_depth_to_color/image_raw',
                '/realsense/aligned_depth_to_color/image_raw',
            ),
        ]

    try:
        realsense_pkg_dir = get_package_share_directory('realsense2_camera')
        realsense_launch_file = os.path.join(realsense_pkg_dir, 'launch', 'rs_launch.py')
    except Exception as e:
        print(f'Warning: realsense2_camera package not found: {e}')
        realsense_launch_file = None

    launch_config = []
    if realsense_launch_file and skip_rs == 'false':
        launch_config.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    'align_depth.enable': 'true',
                    'rgb_camera.profile': '1280x720x30',
                    'depth_module.profile': '848x480x30',
                }.items(),
            )
        )

    detection_manager_node = Node(
        package='computer_vision',
        executable='computer_vision.detection_manager',
        name='detection_manager',
        output='screen',
    )

    detector_node = Node(
        package='computer_vision',
        executable='detector',
        name='detector',
        output='screen',
        remappings=remappings,
    )

    set_mode_service_call = ExecuteProcess(
        cmd=[
            'sleep',
            '5',
            '&&',
            'ros2',
            'service',
            'call',
            '/detection/set_mode',
            'reseq_interfaces/srv/SetMode',
            ['"{mode: ', mode, '}"'],
        ],
        output='screen',
        shell=True,
    )

    launch_config.extend([detection_manager_node, detector_node, set_mode_service_call])
    return launch_config


def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        'mode', default_value='2', description='Detection mode to set'
    )

    skip_realsense_arg = DeclareLaunchArgument(
        'skip_realsense',
        default_value='false',
        description='Skip RealSense launch (set true when sensors_launch already starts it)',
    )

    return LaunchDescription([mode_arg, skip_realsense_arg, OpaqueFunction(function=launch_setup)])
