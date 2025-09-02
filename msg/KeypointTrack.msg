# KeypointTrack.msg
vision_msgs/Detection2D detection  # 基础检测信息
geometry_msgs/Point[] keypoints    # 17个关键点坐标
float32[] scores                   # 17个关键点置信度