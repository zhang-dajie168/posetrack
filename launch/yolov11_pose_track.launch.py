import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import SetEnvironmentVariable

def generate_launch_description():
    # 获取posetrack包的路径
    pkg_posetrack = get_package_share_directory('posetrack')
    
    # 设置模型路径
    model_path = os.path.join(
        pkg_posetrack,
        'models',
        'yolov8n_pose_bayese_640x640_nv12_modified.bin'
    )
    
    osnet_model_path = os.path.join(
        pkg_posetrack,
        'models',
        'osnet_64x128_nv12.bin'
    )
    
    # 设置LD_LIBRARY_PATH
    lib_path = os.path.join(pkg_posetrack, '..', 'lib')
    
    # 声明启动参数
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=model_path,
        description='Path to the YOLOv11 pose model'
    )
    
    osnet_model_path_arg = DeclareLaunchArgument(
        'osnet_model_path',
        default_value=osnet_model_path,
        description='Path to the OSNet re-identification model'
    )
    
    conf_threshold_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.3',
        description='Confidence threshold for detection'
    )
    
    kpt_conf_threshold_arg = DeclareLaunchArgument(
        'kpt_conf_threshold',
        default_value='0.6',
        description='Keypoint confidence threshold'
    )
    
    real_person_height_arg = DeclareLaunchArgument(
        'real_person_height',
        default_value='1.7',
        description='Real person height in meters for depth estimation'
    )
    
    max_processing_fps_arg = DeclareLaunchArgument(
        'max_processing_fps',
        default_value='15',
        description='Maximum processing FPS'
    )
    
    reid_similarity_threshold_arg = DeclareLaunchArgument(
        'reid_similarity_threshold',
        default_value='0.7',
        description='Re-identification similarity threshold'
    )
    
    hands_up_duration_threshold_arg = DeclareLaunchArgument(
        'hands_up_duration_threshold',
        default_value='2.0',
        description='Hands-up duration threshold in seconds'
    )
    
    hands_up_cooldown_arg = DeclareLaunchArgument(
        'hands_up_cooldown',
        default_value='10.0',
        description='Hands-up detection cooldown in seconds'
    )
    
    max_lost_frames_arg = DeclareLaunchArgument(
        'max_lost_frames',
        default_value='10',
        description='Maximum number of frames to keep tracking after target is lost'
    )
    
    # 设置环境变量
    env_vars = SetEnvironmentVariable(
        'LD_LIBRARY_PATH',
        value=lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    )
    
    # 使用标准的Node动作（推荐）
    pose_track_node = Node(
        package='posetrack',
        executable='yolov11_pose_track_node',
        name='yolov11_pose_track',
        output='screen',
        parameters=[
            {'model_path': LaunchConfiguration('model_path')},
            {'osnet_model_path': LaunchConfiguration('osnet_model_path')},
            {'conf_threshold': LaunchConfiguration('conf_threshold')},
            {'kpt_conf_threshold': LaunchConfiguration('kpt_conf_threshold')},
            {'real_person_height': LaunchConfiguration('real_person_height')},
            {'max_processing_fps': LaunchConfiguration('max_processing_fps')},
            {'reid_similarity_threshold': LaunchConfiguration('reid_similarity_threshold')},
            {'hands_up_duration_threshold': LaunchConfiguration('hands_up_duration_threshold')},
            {'hands_up_cooldown': LaunchConfiguration('hands_up_cooldown')},
            {'max_lost_frames': LaunchConfiguration('max_lost_frames')}
        ]
    )
    
    return LaunchDescription([
        env_vars,
        model_path_arg,
        osnet_model_path_arg,
        conf_threshold_arg,
        kpt_conf_threshold_arg,
        real_person_height_arg,
        max_processing_fps_arg,
        reid_similarity_threshold_arg,
        hands_up_duration_threshold_arg,
        hands_up_cooldown_arg,
        max_lost_frames_arg,
        pose_track_node
    ])