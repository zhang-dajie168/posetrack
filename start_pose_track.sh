#!/bin/bash
# start_pose_track_launch.sh

source install/setup.bash

# 设置库路径
export LD_LIBRARY_PATH=/home/sunrise/Ebike_Human_Follower/install/posetrack/lib:$LD_LIBRARY_PATH

# 直接运行节点
ros2 run posetrack yolov11_pose_track_node \
    --ros-args \
    -p model_path:=/home/sunrise/Ebike_Human_Follower/install/posetrack/share/posetrack/models/yolov8n_pose_bayese_640x640_nv12_modified.bin \
    -p osnet_model_path:=/home/sunrise/Ebike_Human_Follower/install/posetrack/share/posetrack/models/osnet_64x128_nv12.bin \
    -p conf_threshold:=0.3 \
    -p kpt_conf_threshold:=0.6 \
    -p real_person_height:=1.7 \
    -p max_processing_fps:=15 \
    -p reid_similarity_threshold:=0.7 \
    -p hands_up_duration_threshold:=10.0 \
    -p hands_up_cooldown:=10.0 \
    -p max_lost_frames:=10
    
