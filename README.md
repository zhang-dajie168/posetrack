# posetrack


## 目录结构
posetrack/
├── launch/
│   └── yolov11_pose_track.launch.py
├── msg/
│   └── KeypointTrack.msg
├── models/
│   └── yolo11n_pose_bayese_640x640_nv12_modified.bin
├── src/
│   └── yolov11_pose_track.cc
│   └── bytetrack
├── CMakeLists.txt
└── package.xml


# sudo apt-get install ros-humble-vision-msgs
## sudo apt-get install libpcl-dev ros-humble-pcl-conversions ros-humble-pcl-ros


## build
cd ~/Ebike_Human_Follower

 colcon build --packages-select posetrack


## run
 source install/setup.bash

 ./start_pose_track.sh




## 查看话题 
ros2 topic list

/person_positions


## 查看话题 输出内容
ros2 topic echo /person_positions

header:
  stamp:
    sec: 1752474143
    nanosec: 893721088
  frame_id: camera_link
point:
  x: 357.22271728515625
  y: 307.40899658203125
  z: 0.7892728447914124



<!-- ## 输出检测结果说明:

时间戳: 1752137165.763322112 秒
坐标系: camera_color_optical_frame
边界框中心位置: (413.10, 253.02)
边界框尺寸: 244.25 × 224.02 像素
跟踪ID: '0'


    kpt_msg->detection.id     //检测ID

    Set bounding box info  （左上角x,左上角y,box的宽size_x，box的高size_y）

    kpt_msg->detection.bbox.center.position.x           
    kpt_msg->detection.bbox.center.position.y 
    kpt_msg->detection.bbox.size_x 
    kpt_msg->detection.bbox.size_y 
    kpt_msg->detection.bbox.center.theta = 0;   //目标物体的深度

    kpt_msg->keypoints    //17个关键点（0-16）
    kpt_msg->scores       //17个关键点对应的置信度（0-16） -->


<!-- ## ros2 topic echo /keypoint_tracks
  
detection:
  header:
    stamp:
      sec: 1752219496
      nanosec: 261725952
    frame_id: camera_color_optical_frame
  results: []
  bbox:
    center:
      position:
        x: 405.2229309082031     
        y: 288.5151062011719
      theta: 0.7619090676307678         //關鍵點的平均距離
    size_x: 419.29266357421875
    size_y: 359.000732421875
  id: '1' -->



<!-- ## 人体关键点对应身体位置说明 ，以下是标准17点人体姿态关键点模型的身体部位对应关系：
关键点索引	身体部位	说明
0	鼻子	人脸正中的鼻子位置
1	左眼	人脸的左眼中心位置
2	右眼	人脸的右眼中心位置
3	左耳	左耳的位置
4	右耳	右耳的位置
5	左肩	左肩关节位置
6	右肩	右肩关节位置
7	左肘	左肘关节位置
8	右肘	右肘关节位置
9	左手腕	左手腕关节位置
10	右手腕	右手腕关节位置
11	左髋	左髋关节(骨盆左侧)位置
12	右髋	右髋关节(骨盆右侧)位置
13	左膝	左膝关节位置
14	右膝	右膝关节位置
15	左踝	左脚踝关节位置
16	右踝	右脚踝关节位置 -->
