/*
20250901 优化版本 - 性能优化版
举手检测开始跟踪，冷却时间10秒，id丢失通过osnet找回
修复目标丢失后无法重新识别的问题
添加异步处理优化帧率稳定性

// 保存举手目标信息 - 修改为只保存一次
// 更新已保存目标的信息 - 修改为只在ID切换时更新
// 处理目标跟踪逻辑 - 修改调用update_saved_target的方式
// 优化ReID匹配方法，减少不必要的特征提取
// 优化举手检测方法，减少计算量
// 优化可视化方法，减少字符串操作

*/

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "yolov11_pose.h"
#include "bytetrack/BYTETracker.h"
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <mutex>
#include <deque>
#include <algorithm>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <numeric>
#include <chrono>
#include <thread>
#include <atomic>
#include "posetrack/msg/keypoint_track.hpp"
#include "OSNet.h"

class DepthFilter
{
public:
    DepthFilter(size_t window_size = 5, float threshold = 0.5f)
        : window_size_(window_size), threshold_(threshold) {}

    void add_depth(float depth)
    {
        if (depths_.size() >= window_size_)
        {
            depths_.pop_front();
        }
        depths_.push_back(depth);
    }

    float get_filtered_depth()
    {
        if (depths_.empty())
            return 0.0f;

        if (depths_.size() > 1)
        {
            float diff = std::abs(depths_.back() - depths_[depths_.size() - 2]);
            if (diff > threshold_)
            {
                return depths_[depths_.size() - 2];
            }
        }

        if (depths_.size() > 1)
        {
            std::vector<float> sorted(depths_.begin(), depths_.end());
            std::sort(sorted.begin(), sorted.end());
            return sorted[sorted.size() / 2];
        }
        return depths_.back();
    }

private:
    std::deque<float> depths_;
    size_t window_size_;
    float threshold_;
};

// 简化的跟踪目标信息结构体
struct TrackedTarget
{
    int track_id;
    cv::Rect bbox;
    std::vector<float> reid_features; // OSNet提取的特征向量
    rclcpp::Time last_update_time;    // 上次更新时间
    rclcpp::Time hands_up_start_time; // 举手开始时间
    bool hands_up_confirmed;          // 举手已确认
    int lost_frames;                  // 丢失帧数计数器

    TrackedTarget() : track_id(-1),
                      last_update_time(rclcpp::Time(0, 0, RCL_ROS_TIME)),
                      hands_up_start_time(rclcpp::Time(0, 0, RCL_ROS_TIME)),
                      hands_up_confirmed(false),
                      lost_frames(0) {}
};

class Yolov11PoseNode : public rclcpp::Node
{
public:
    Yolov11PoseNode() : Node("yolov11_pose_node"),
                        last_process_time_(this->now()),
                        last_hands_up_detection_time_(rclcpp::Time(0, 0, RCL_ROS_TIME)),
                        last_perf_log_time_(this->now()),
                        last_tracking_log_time_(this->now()),
                        processing_(false),
                        stop_processing_(false)
    {
        // Camera intrinsics (Orbbec Gemini 335L 640x480)
        fx_ = 367.21f;
        fy_ = 316.44f;
        cx_ = 367.20f;
        cy_ = 244.60f;

        // Declare parameters
        this->declare_parameter<std::string>("model_path", "");
        this->declare_parameter<std::string>("osnet_model_path", "");
        this->declare_parameter<float>("conf_threshold", 0.35f);
        this->declare_parameter<float>("kpt_conf_threshold", 0.5f);
        this->declare_parameter<float>("real_person_height", 1.7f);
        this->declare_parameter<int>("max_processing_fps", 15); // 提高到15Hz
        this->declare_parameter<float>("reid_similarity_threshold", 0.7f);
        this->declare_parameter<float>("hands_up_duration_threshold", 10.0f);
        this->declare_parameter<float>("hands_up_cooldown", 10.0f);
        this->declare_parameter<int>("max_lost_frames", 10); // 最大丢失帧数

        // Get parameters
        std::string model_path = this->get_parameter("model_path").as_string();
        std::string osnet_model_path = this->get_parameter("osnet_model_path").as_string();
        conf_threshold_ = this->get_parameter("conf_threshold").as_double();
        kpt_conf_threshold_ = this->get_parameter("kpt_conf_threshold").as_double();
        real_person_height_ = this->get_parameter("real_person_height").as_double();
        max_processing_fps_ = this->get_parameter("max_processing_fps").as_int();
        reid_similarity_threshold_ = this->get_parameter("reid_similarity_threshold").as_double();
        hands_up_duration_threshold_ = this->get_parameter("hands_up_duration_threshold").as_double();
        hands_up_cooldown_ = this->get_parameter("hands_up_cooldown").as_double();
        max_lost_frames_ = this->get_parameter("max_lost_frames").as_int();
        min_process_interval_ = 1.0 / max_processing_fps_;

        // Initialize pose detection model
        pose_detector_ = std::make_unique<YOLOv11Pose>(model_path, 1);
        if (!pose_detector_->init_model())
        {
            RCLCPP_ERROR(this->get_logger(), "Pose model initialization failed");
            rclcpp::shutdown();
            return;
        }

        // Initialize OSNet re-identification model
        if (!osnet_model_path.empty())
        {
            osnet_ = std::make_unique<OSNetRDKX5Inference>(osnet_model_path, cv::Size(64, 128));
            if (!osnet_->init_model())
            {
                RCLCPP_ERROR(this->get_logger(), "OSNet model initialization failed");
                rclcpp::shutdown();
                return;
            }
            RCLCPP_INFO(this->get_logger(), "OSNet model loaded successfully");
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "No OSNet model path provided, re-identification disabled");
        }

        // Calculate raw confidence thresholds
        conf_threshold_raw_ = -log(1 / conf_threshold_ - 1);
        kpt_conf_threshold_raw_ = -log(1 / kpt_conf_threshold_ - 1);

        // Create subscribers with QoS settings
        rclcpp::QoS image_qos(10);
        image_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", image_qos,
            std::bind(&Yolov11PoseNode::image_callback, this, std::placeholders::_1));

        depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", image_qos,
            std::bind(&Yolov11PoseNode::depth_image_callback, this, std::placeholders::_1));

        // Create publishers
        person_point_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/person_positions", 10);
        detect_pose_pub_ = this->create_publisher<sensor_msgs::msg::Image>("detect_pose", 10);
        keypoint_tracks_pub_ = this->create_publisher<posetrack::msg::KeypointTrack>("/keypoint_tracks", 10);

        // Initialize tracker
        tracker_ = std::make_unique<BYTETracker>(30, 30);

        // Initialize skeleton connections
        const int init_skeleton[36] = {
            // 身体主干
            6, 5,   // 左肩-右肩
            5, 11,  // 右肩-右胯
            6, 12,  // 左肩-左胯
            11, 12, // 右胯-左胯
            11, 13, // 右胯-右膝
            12, 14, // 左胯-左膝
            13, 15, // 右膝-右脚踝
            14, 16, // 左膝-左脚踝

            // 手臂
            5, 7,  // 右肩-右肘
            7, 9,  // 右肘-右手腕
            6, 8,  // 左肩-左肘
            8, 10, // 左肘-左手腕

            // 面部（可选）
            1, 2, // 鼻子-左眼
            1, 3, // 鼻子-右眼
            2, 4, // 左眼-右眼
            0, 1, // 头部中心-鼻子

            // 额外连接确保正确性
            5, 6,  // 右肩-左肩
            11, 12 // 右胯-左胯
        };
        memcpy(skeleton_, init_skeleton, sizeof(init_skeleton));

        // 添加性能统计变量
        total_frames_ = 0;
        total_processing_time_ = 0.0;

        // 启动处理线程
        processing_thread_ = std::thread(&Yolov11PoseNode::processing_loop, this);

        RCLCPP_INFO(this->get_logger(), "YOLOv11 Pose Tracking node initialized (Max FPS: %d)", max_processing_fps_);
        RCLCPP_INFO(this->get_logger(), "Hands-up detection: %.1fs threshold, %.1fs cooldown",
                    hands_up_duration_threshold_, hands_up_cooldown_);
    }

    ~Yolov11PoseNode()
    {
        stop_processing_ = true;
        if (processing_thread_.joinable())
        {
            processing_thread_.join();
        }
    }

private:
    struct TrackedPerson
    {
        int track_id;
        bool is_tracking;
        bool hands_up;
        rclcpp::Time last_hands_up_time;
        rclcpp::Time hands_up_start_time;
        bool hands_up_confirmed;

        TrackedPerson() : track_id(0), is_tracking(false), hands_up(false),
                          last_hands_up_time(rclcpp::Time(0, 0, RCL_ROS_TIME)),
                          hands_up_start_time(rclcpp::Time(0, 0, RCL_ROS_TIME)),
                          hands_up_confirmed(false) {}
    };

    enum class TrackingState
    {
        IDLE,     // 空闲状态，等待检测举手
        TRACKING, // 正在跟踪目标
        COOLDOWN  // 冷却状态，停止跟踪后等待
    };

    // 添加性能计时结构体
    struct ProcessingTimes
    {
        double image_conversion = 0.0;
        double preprocessing = 0.0;
        double inference = 0.0;
        double postprocessing = 0.0;
        double tracking = 0.0;
        double visualization = 0.0;
        double publishing = 0.0;
        double total = 0.0;
    };

    float estimate_depth_from_bbox_height(float bbox_height_pixels)
    {
        return (real_person_height_ * fy_) / bbox_height_pixels;
    }

    float get_keypoints_depth(const STrack &track)
    {
        std::vector<float> valid_depths;
        const std::vector<int> keypoint_indices = {5, 6, 11, 12}; // Shoulders and hips

        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (depth_image_.empty())
            return 0.0f;

        for (int idx : keypoint_indices)
        {
            if (static_cast<size_t>(idx) >= track.keypoints.size())
                continue;

            float x = track.keypoints[idx].x;
            float y = track.keypoints[idx].y;

            if (x < 0 || y < 0 || x >= depth_image_.cols || y >= depth_image_.rows)
                continue;

            float depth = depth_image_.at<float>(y, x) / 1000.0f;
            if (depth > 0.5f && depth < 8.0f)
            {
                valid_depths.push_back(depth);
            }
        }

        if (valid_depths.empty())
            return 0.0f;

        // Calculate median
        std::sort(valid_depths.begin(), valid_depths.end());
        return valid_depths[valid_depths.size() / 2];
    }

    float compute_body_depth(const STrack &track)
    {
        // 1. Get depth from keypoints
        float keypoints_depth = get_keypoints_depth(track);

        // 2. Estimate depth from bbox height
        float bbox_height = track.tlbr[3] - track.tlbr[1];
        float bbox_estimated_depth = estimate_depth_from_bbox_height(bbox_height);

        // 3. Fusion strategy
        float final_depth;
        if (keypoints_depth <= 0.1f)
        {
            final_depth = bbox_estimated_depth;
        }
        else
        {
            // Check for depth jumps (>1m difference)
            if (std::abs(keypoints_depth - bbox_estimated_depth) > 1.0f)
            {
                final_depth = bbox_estimated_depth;
            }
            else
            {
                // Weighted fusion (70% keypoints, 30% bbox)
                final_depth = keypoints_depth * 0.7f + bbox_estimated_depth * 0.3f;
            }
        }

        // 4. Apply depth filter
        auto &depth_filter = depth_filters_[track.track_id];
        depth_filter.add_depth(final_depth);
        return depth_filter.get_filtered_depth();
    }

    // 从图像中裁剪目标区域用于ReID特征提取
    cv::Mat crop_target_region(const cv::Mat &image, const STrack &track)
    {
        int x1 = static_cast<int>(track.tlbr[0]);
        int y1 = static_cast<int>(track.tlbr[1]);
        int x2 = static_cast<int>(track.tlbr[2]);
        int y2 = static_cast<int>(track.tlbr[3]);

        // 确保边界在图像范围内
        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(image.cols - 1, x2);
        y2 = std::min(image.rows - 1, y2);

        // 裁剪目标区域
        return image(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    }

    // 保存举手目标信息 - 只保存一次特征
    void save_hands_up_target(const cv::Mat &image, const STrack &track)
    {
        if (!osnet_)
            return;

        // 只在第一次保存时提取特征
        tracked_target_.track_id = track.track_id;
        tracked_target_.bbox = cv::Rect(
            static_cast<int>(track.tlbr[0]),
            static_cast<int>(track.tlbr[1]),
            static_cast<int>(track.tlbr[2] - track.tlbr[0]),
            static_cast<int>(track.tlbr[3] - track.tlbr[1]));
        tracked_target_.last_update_time = this->now();
        tracked_target_.hands_up_start_time = this->now();
        tracked_target_.hands_up_confirmed = true;
        tracked_target_.lost_frames = 0;

        // 只在特征为空时才提取ReID特征（确保只保存一次）
        if (tracked_target_.reid_features.empty())
        {
            try
            {
                cv::Mat target_roi = crop_target_region(image, track);
                if (!target_roi.empty())
                {
                    tracked_target_.reid_features = osnet_->extract_features(target_roi);
                    RCLCPP_INFO(this->get_logger(), "Saved target %d with ReID features size: %zu",
                                tracked_target_.track_id, tracked_target_.reid_features.size());
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to extract ReID features: %s", e.what());
            }
        }
        else
        {
            RCLCPP_DEBUG(this->get_logger(), "Target %d features already exist, skipping extraction",
                         tracked_target_.track_id);
        }
    }

    // 清空跟踪目标
    void clear_tracked_target()
    {
        tracked_target_.track_id = -1;
        tracked_target_.reid_features.clear(); // 清空特征
        tracked_target_.hands_up_confirmed = false;
        tracked_target_.hands_up_start_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
        tracked_target_.lost_frames = 0;
        RCLCPP_INFO(this->get_logger(), "Tracked target cleared");
    }

    // 更新已保存目标的信息 - 只更新位置信息，不更新特征
    void update_saved_target(const STrack &track)
    {
        if (tracked_target_.track_id == -1)
            return;

        tracked_target_.bbox = cv::Rect(
            static_cast<int>(track.tlbr[0]),
            static_cast<int>(track.tlbr[1]),
            static_cast<int>(track.tlbr[2] - track.tlbr[0]),
            static_cast<int>(track.tlbr[3] - track.tlbr[1]));
        tracked_target_.last_update_time = this->now();
        tracked_target_.lost_frames = 0; // 重置丢失帧数
    }

    // 重新识别目标 - 使用保存的特征进行匹配
    int reidentify_target(const cv::Mat &image, const std::vector<STrack> &tracks)
    {
        if (!osnet_ || tracked_target_.track_id == -1 || tracked_target_.reid_features.empty())
        {
            RCLCPP_DEBUG(this->get_logger(), "ReID条件不满足: osnet=%d, target_id=%d, features_empty=%d",
                         (osnet_ != nullptr), tracked_target_.track_id, tracked_target_.reid_features.empty());
            return -1;
        }

        int best_match_id = -1;
        float best_similarity = reid_similarity_threshold_;

        RCLCPP_INFO(this->get_logger(), "开始ReID匹配，使用保存的目标 %d 特征", tracked_target_.track_id);

        // 预筛选：只处理与目标bbox大小相似的track
        for (const auto &track : tracks)
        {
            // 快速筛选：bbox面积差异过大则跳过
            float target_area = tracked_target_.bbox.width * tracked_target_.bbox.height;
            float current_area = (track.tlbr[2] - track.tlbr[0]) * (track.tlbr[3] - track.tlbr[1]);
            float area_ratio = std::max(target_area, current_area) / std::min(target_area, current_area);

            if (area_ratio > 2.0f) // 面积差异超过2倍则跳过
            {
                continue;
            }

            try
            {
                // 提取当前跟踪目标的特征
                cv::Mat current_roi = crop_target_region(image, track);
                if (current_roi.empty())
                {
                    continue;
                }

                auto current_features = osnet_->extract_features(current_roi);
                if (current_features.empty())
                {
                    continue;
                }

                // 使用保存的特征计算相似度（不更新保存的特征）
                float similarity = osnet_->compute_similarity(tracked_target_.reid_features, current_features);

                RCLCPP_DEBUG(this->get_logger(), "Track %d 与保存目标 %d 的相似度: %.3f (阈值: %.3f)",
                             track.track_id, tracked_target_.track_id, similarity, reid_similarity_threshold_);

                if (similarity > best_similarity)
                {
                    best_similarity = similarity;
                    best_match_id = track.track_id;
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_DEBUG(this->get_logger(), "Track %d ReID特征提取失败: %s", track.track_id, e.what());
            }
        }

        if (best_match_id != -1)
        {
            RCLCPP_INFO(this->get_logger(), "ReID匹配成功: 保存目标 %d -> 当前track %d, 相似度: %.3f",
                        tracked_target_.track_id, best_match_id, best_similarity);
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "ReID匹配失败: 未找到相似目标");
        }

        return best_match_id;
    }

    // 图像回调 - 简化版本，只做入队操作
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        // 简化回调，只做入队操作
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            // 限制队列大小，避免内存溢出
            if (image_queue_.size() < 5)
            {
                image_queue_.push_back(msg);
            }
            else
            {
                RCLCPP_DEBUG(this->get_logger(), "Queue full, dropping frame");
            }
        }
    }

    // 处理循环线程
    void processing_loop()
    {
        while (!stop_processing_)
        {
            if (!process_next_frame())
            {
                // 没有帧需要处理，短暂休眠
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }

    // 处理下一帧
    bool process_next_frame()
    {
        sensor_msgs::msg::Image::ConstSharedPtr msg;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (image_queue_.empty())
            {
                return false;
            }
            msg = image_queue_.front();
            image_queue_.pop_front();
        }

        process_single_frame(msg);
        return true;
    }

    // 处理单帧图像
    void process_single_frame(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        auto start_total = std::chrono::high_resolution_clock::now();
        ProcessingTimes times;

        try
        {
            // 图像转换计时
            auto start_conv = std::chrono::high_resolution_clock::now();
            cv_bridge::CvImageConstPtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            }
            catch (cv_bridge::Exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }
            auto end_conv = std::chrono::high_resolution_clock::now();
            times.image_conversion = std::chrono::duration<double>(end_conv - start_conv).count();

            const cv::Mat &img = cv_ptr->image;
            if (img.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Received empty image");
                return;
            }

            // 预处理计时
            auto start_preprocess = std::chrono::high_resolution_clock::now();
            cv::Mat processed_img;
            current_x_shift_int_ = static_cast<int>(current_x_shift_);
            current_y_shift_int_ = static_cast<int>(current_y_shift_);

            if (!pose_detector_->preprocess_image(img, processed_img,
                                                  current_x_scale_, current_y_scale_,
                                                  current_x_shift_int_, current_y_shift_int_))
            {
                RCLCPP_ERROR(this->get_logger(), "Image preprocessing failed");
                return;
            }
            auto end_preprocess = std::chrono::high_resolution_clock::now();
            times.preprocessing = std::chrono::duration<double>(end_preprocess - start_preprocess).count();

            // 推理计时
            auto start_inference = std::chrono::high_resolution_clock::now();
            bool inference_success = pose_detector_->run_inference(processed_img);
            auto end_inference = std::chrono::high_resolution_clock::now();
            times.inference = std::chrono::duration<double>(end_inference - start_inference).count();

            // 后处理计时
            auto start_postprocess = std::chrono::high_resolution_clock::now();
            bool postprocess_success = false;
            if (inference_success)
            {
                postprocess_success = pose_detector_->postprocess(conf_threshold_raw_, kpt_conf_threshold_raw_);
            }
            auto end_postprocess = std::chrono::high_resolution_clock::now();
            times.postprocessing = std::chrono::duration<double>(end_postprocess - start_postprocess).count();

            if (!inference_success || !postprocess_success)
            {
                RCLCPP_ERROR(this->get_logger(), "Inference or postprocessing failed");
                return;
            }

            // 跟踪计时
            auto start_tracking = std::chrono::high_resolution_clock::now();
            const auto &detections = pose_detector_->get_detections();
            const auto &nms_indices = pose_detector_->get_nms_indices();
            auto trackobj = convert_detections_to_trackobj(detections, nms_indices);
            auto tracks = tracker_->update(trackobj);

            update_tracking_states(tracks);
            process_target_tracking(img, tracks);
            auto end_tracking = std::chrono::high_resolution_clock::now();
            times.tracking = std::chrono::duration<double>(end_tracking - start_tracking).count();

            // 可视化计时
            auto start_visualization = std::chrono::high_resolution_clock::now();
            if (detect_pose_pub_->get_subscription_count() > 0 && total_frames_ % 2 == 0) // 每2帧可视化一次
            {
                cv::Mat result_img = img.clone();
                visualize_results(result_img, tracks);
                auto result_msg = cv_bridge::CvImage(msg->header, "bgr8", result_img).toImageMsg();

                auto start_publish = std::chrono::high_resolution_clock::now();
                detect_pose_pub_->publish(*result_msg);
                auto end_publish = std::chrono::high_resolution_clock::now();
                times.publishing += std::chrono::duration<double>(end_publish - start_publish).count();
            }
            auto end_visualization = std::chrono::high_resolution_clock::now();
            times.visualization = std::chrono::duration<double>(end_visualization - start_visualization).count();

            // 发布其他消息的计时
            auto start_other_publish = std::chrono::high_resolution_clock::now();
            publish_person_positions(msg->header, tracks);
            publish_tracked_keypoints(msg->header, tracks);
            auto end_other_publish = std::chrono::high_resolution_clock::now();
            times.publishing += std::chrono::duration<double>(end_other_publish - start_other_publish).count();

            // 性能统计
            auto end_total = std::chrono::high_resolution_clock::now();
            times.total = std::chrono::duration<double>(end_total - start_total).count();

            // 更新统计信息
            total_frames_++;
            total_processing_time_ += times.total;

            // 每30帧打印一次性能信息（减少输出频率）
            if (total_frames_ % 30 == 0)
            {
                print_performance_stats(times, msg->header);
            }

            // 打印跟踪信息
            print_tracking_info(tracks, total_frames_);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Processing error: %s", e.what());
        }
    }

    // 处理目标跟踪逻辑 - 修改为不更新特征
    void process_target_tracking(const cv::Mat &image, const std::vector<STrack> &tracks)
    {
        auto current_time = this->now();

        switch (tracking_state_)
        {
        case TrackingState::IDLE:
            // 空闲状态，检测举手开始跟踪
            if (tracked_target_.track_id == -1)
            {
                for (const auto &track : tracks)
                {
                    if (isHandsUp(track))
                    {
                        // 检测到举手，开始跟踪并保存特征
                        save_hands_up_target(image, track);
                        tracking_state_ = TrackingState::TRACKING;
                        tracking_start_time_ = current_time;
                        RCLCPP_INFO(this->get_logger(), "Tracking started for target %d", track.track_id);
                        return;
                    }
                }
            }
            break;

        case TrackingState::TRACKING:
            // 跟踪状态，检查是否达到10秒并检测到举手停止
            if (tracked_target_.track_id != -1)
            {
                double tracking_duration = (current_time - tracking_start_time_).seconds();

                // 检查当前目标是否在跟踪列表中
                bool target_found = false;
                int found_track_id = -1;

                for (const auto &track : tracks)
                {
                    if (track.track_id == tracked_target_.track_id)
                    {
                        target_found = true;
                        found_track_id = track.track_id;

                        // 如果跟踪时间超过10秒且检测到举手，停止跟踪
                        if (tracking_duration >= 10.0 && isHandsUp(track))
                        {
                            clear_tracked_target();
                            tracking_state_ = TrackingState::COOLDOWN;
                            cooldown_start_time_ = current_time;
                            RCLCPP_INFO(this->get_logger(), "Tracking stopped after %.1f seconds", tracking_duration);
                            return;
                        }

                        // 只更新位置信息，不更新特征
                        update_saved_target(track);
                        break;
                    }
                }

                // 如果目标丢失，尝试重新识别
                if (!target_found)
                {
                    tracked_target_.lost_frames++;
                    RCLCPP_WARN(this->get_logger(), "Target %d lost, lost frames: %d/%d",
                                tracked_target_.track_id, tracked_target_.lost_frames, max_lost_frames_);

                    // 如果丢失帧数超过阈值，尝试重新识别
                    if (tracked_target_.lost_frames >= max_lost_frames_)
                    {
                        RCLCPP_INFO(this->get_logger(), "Target %d lost for too long, attempting re-identification",
                                    tracked_target_.track_id);
                        int reidentified_id = reidentify_target(image, tracks);

                        if (reidentified_id != -1)
                        {
                            // 找到匹配的目标，只更新ID和位置信息，不更新特征
                            for (const auto &track : tracks)
                            {
                                if (track.track_id == reidentified_id)
                                {
                                    tracked_target_.track_id = reidentified_id;
                                    update_saved_target(track); // 只更新位置，不更新特征
                                    RCLCPP_INFO(this->get_logger(), "Target reidentified: %d (features preserved)",
                                                reidentified_id);
                                    break;
                                }
                            }
                        }
                        else
                        {
                            // 重新识别失败，继续使用保存的特征
                            RCLCPP_INFO(this->get_logger(), "Target %d not found, keeping saved features",
                                        tracked_target_.track_id);
                        }
                    }
                }
            }
            break;

        case TrackingState::COOLDOWN:
            // 冷却状态，等待10秒后回到空闲状态
            double cooldown_duration = (current_time - cooldown_start_time_).seconds();
            if (cooldown_duration >= 10.0)
            {
                tracking_state_ = TrackingState::IDLE;
                RCLCPP_INFO(this->get_logger(), "Cooldown finished, ready for new tracking");
            }
            break;
        }
    }
    void publish_tracked_keypoints(const std_msgs::msg::Header &header, const std::vector<STrack> &tracks)
    {
        // 只发布保存的目标
        if (tracked_target_.track_id == -1)
            return;

        // 查找当前帧中的目标
        const STrack *current_track = nullptr;
        for (const auto &track : tracks)
        {
            if (track.track_id == tracked_target_.track_id)
            {
                current_track = &track;
                break;
            }
        }

        // 如果目标在当前帧中不存在，跳过发布
        if (!current_track)
            return;

        auto kpt_msg = std::make_shared<posetrack::msg::KeypointTrack>();
        kpt_msg->detection.header = header;
        kpt_msg->detection.header.frame_id = "camera_link";
        kpt_msg->detection.id = std::to_string(current_track->track_id);

        // 获取身体深度（实时计算）
        // float current_depth = compute_body_depth(*current_track);
        kpt_msg->detection.bbox.center.theta = 0;

        // 计算边界框中心点
        float center_x = current_track->tlbr[0];
        float center_y = current_track->tlbr[1];

        // 设置边界框信息
        kpt_msg->detection.bbox.center.position.x = center_x;
        kpt_msg->detection.bbox.center.position.y = center_y;

        // 边界框尺寸转换
        float bbox_width = current_track->tlbr[2];
        float bbox_height = current_track->tlbr[3];
        kpt_msg->detection.bbox.size_x = bbox_width;
        kpt_msg->detection.bbox.size_y = bbox_height;

        // 设置关键点和分数
        for (size_t i = 0; i < current_track->keypoints.size(); ++i)
        {
            geometry_msgs::msg::Point point;
            float img_x = current_track->keypoints[i].x;
            float img_y = current_track->keypoints[i].y;

            point.x = img_x;
            point.y = img_y;
            point.z = 0;

            kpt_msg->keypoints.push_back(point);
            kpt_msg->scores.push_back(current_track->keypoints_score[i]);
        }

        keypoint_tracks_pub_->publish(*kpt_msg);
    }

    void publish_person_positions(const std_msgs::msg::Header &header, const std::vector<STrack> &tracks)
    {
        // 只发布保存的目标
        if (tracked_target_.track_id == -1)
            return;

        // 查找当前帧中的目标
        const STrack *current_track = nullptr;
        for (const auto &track : tracks)
        {
            if (track.track_id == tracked_target_.track_id)
            {
                current_track = &track;
                break;
            }
        }

        // 如果目标在当前帧中不存在，跳过发布
        if (!current_track)
            return;

        // 计算中心点坐标
        float center_x = current_track->tlbr[0] + (current_track->tlbr[2] - current_track->tlbr[0]) / 2.0f;
        float center_y = current_track->tlbr[1] + (current_track->tlbr[3] - current_track->tlbr[1]) / 2.0f;

        // 获取实时深度
        float current_depth = compute_body_depth(*current_track);

        // 转换为相机坐标系
        float cam_x, cam_y, cam_z;
        imageToCameraCoords(center_x, center_y, current_depth, cam_x, cam_y, cam_z);

        auto point_msg = geometry_msgs::msg::PointStamped();
        point_msg.header = header;
        point_msg.header.frame_id = "camera_link";
        point_msg.point.x = cam_x;
        point_msg.point.y = cam_y;
        point_msg.point.z = cam_z;

        person_point_pub_->publish(point_msg);
    }

    void update_tracking_states(const std::vector<STrack> &tracks)
    {
        auto current_time = this->now();

        // 更新所有跟踪人员的状态
        for (const auto &track : tracks)
        {
            int track_id = track.track_id;

            // 初始化或获取跟踪人员
            if (tracked_persons_.find(track_id) == tracked_persons_.end())
            {
                tracked_persons_[track_id] = TrackedPerson();
                tracked_persons_[track_id].track_id = track_id;
                depth_filters_[track_id] = DepthFilter();
            }

            auto &person = tracked_persons_[track_id];

            // 更新举手状态
            bool current_hands_up = isHandsUp(track);

            if (current_hands_up && !person.hands_up)
            {
                // 开始举手，记录开始时间
                person.hands_up_start_time = current_time;
            }
            else if (!current_hands_up && person.hands_up)
            {
                // 停止举手，重置计时器和确认状态
                person.hands_up_start_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
                person.hands_up_confirmed = false;
            }

            person.hands_up = current_hands_up;
            person.last_hands_up_time = current_hands_up ? current_time : person.last_hands_up_time;

            // 简单的跟踪状态管理
            person.is_tracking = (tracked_target_.track_id == track_id);
        }

        // 清理长时间未出现的跟踪人员
        std::vector<int> to_remove;
        for (auto &[track_id, person] : tracked_persons_)
        {
            bool found = false;
            for (const auto &track : tracks)
            {
                if (track.track_id == track_id)
                {
                    found = true;
                    break;
                }
            }

            if (!found && (current_time - person.last_hands_up_time).seconds() > 10.0)
            {
                to_remove.push_back(track_id);
            }
        }

        for (int track_id : to_remove)
        {
            tracked_persons_.erase(track_id);
            depth_filters_.erase(track_id);
        }
    }

    void depth_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        try
        {
            depth_image_ = cv_bridge::toCvShare(msg, "32FC1")->image.clone();
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge Error: %s", e.what());
        }
    }

    void imageToCameraCoords(float img_x, float img_y, float depth,
                             float &cam_x, float &cam_y, float &cam_z)
    {
        cam_z = depth;
        cam_x = (img_x - cx_) * cam_z / fx_;
        cam_y = (img_y - cy_) * cam_z / fy_;
    }

    // 优化举手检测方法，减少计算量
    bool isHandsUp(const STrack &track)
    {
        const int LEFT_WRIST = 9;
        const int RIGHT_WRIST = 10;
        const int LEFT_SHOULDER = 5;
        const int RIGHT_SHOULDER = 6;
        const int NOSE = 0;

        // 快速检查：必须有肩膀和鼻子关键点
        if (LEFT_SHOULDER >= track.keypoints.size() || RIGHT_SHOULDER >= track.keypoints.size() ||
            NOSE >= track.keypoints.size())
        {
            return false;
        }

        // 检查关键点置信度
        if (track.keypoints_score[LEFT_SHOULDER] < kpt_conf_threshold_ ||
            track.keypoints_score[RIGHT_SHOULDER] < kpt_conf_threshold_ ||
            track.keypoints_score[NOSE] < kpt_conf_threshold_)
        {
            return false;
        }

        cv::Point left_shoulder(track.keypoints[LEFT_SHOULDER].x, track.keypoints[LEFT_SHOULDER].y);
        cv::Point right_shoulder(track.keypoints[RIGHT_SHOULDER].x, track.keypoints[RIGHT_SHOULDER].y);
        cv::Point nose(track.keypoints[NOSE].x, track.keypoints[NOSE].y);

        // 检查左手举手
        bool left_hand_up = false;
        if (LEFT_WRIST < track.keypoints.size() && track.keypoints_score[LEFT_WRIST] >= kpt_conf_threshold_)
        {
            cv::Point left_wrist(track.keypoints[LEFT_WRIST].x, track.keypoints[LEFT_WRIST].y);
            left_hand_up = (left_wrist.y < left_shoulder.y) &&
                           (left_wrist.y < nose.y) &&
                           (std::abs(left_wrist.x - left_shoulder.x) < 100);
        }

        // 检查右手举手
        bool right_hand_up = false;
        if (RIGHT_WRIST < track.keypoints.size() && track.keypoints_score[RIGHT_WRIST] >= kpt_conf_threshold_)
        {
            cv::Point right_wrist(track.keypoints[RIGHT_WRIST].x, track.keypoints[RIGHT_WRIST].y);
            right_hand_up = (right_wrist.y < right_shoulder.y) &&
                            (right_wrist.y < nose.y) &&
                            (std::abs(right_wrist.x - right_shoulder.x) < 100);
        }

        return left_hand_up || right_hand_up;
    }

    // 优化可视化方法，减少字符串操作
    void visualize_results(cv::Mat &img, const std::vector<STrack> &tracks)
    {
        static const cv::Scalar active_color(0, 255, 0);
        static const cv::Scalar saved_target_color(255, 0, 0);
        static thread_local std::string label_buffer; // 重用字符串缓冲区

        for (const auto &track : tracks)
        {
            cv::Scalar bbox_color = (tracked_target_.track_id == track.track_id) ? saved_target_color : active_color;

            // 绘制边界框
            cv::Rect bbox(track.tlbr[0], track.tlbr[1], track.tlbr[2] - track.tlbr[0], track.tlbr[3] - track.tlbr[1]);
            cv::rectangle(img, bbox, bbox_color, 2);

            // 重用字符串缓冲区
            label_buffer = "ID: " + std::to_string(track.track_id);
            if (isHandsUp(track))
            {
                label_buffer += " (Hands Up)";
            }

            cv::putText(img, label_buffer, cv::Point(bbox.x, bbox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2);
        }

        // 显示当前跟踪状态
        static thread_local std::string status_buffer;
        switch (tracking_state_)
        {
        case TrackingState::IDLE:
            status_buffer = "State: IDLE - Waiting for hands up";
            break;
        case TrackingState::TRACKING:
            status_buffer = "State: TRACKING - Target ID " + std::to_string(tracked_target_.track_id);
            if (tracked_target_.track_id != -1)
            {
                double tracking_time = (this->now() - tracking_start_time_).seconds();
                status_buffer += " | Time: " + std::to_string(static_cast<int>(tracking_time)) + "s";
            }
            break;
        case TrackingState::COOLDOWN:
            status_buffer = "State: COOLDOWN";
            double cooldown_remaining = 10.0 - (this->now() - cooldown_start_time_).seconds();
            if (cooldown_remaining > 0)
            {
                status_buffer += " | Remaining: " + std::to_string(static_cast<int>(cooldown_remaining)) + "s";
            }
            break;
        }

        cv::putText(img, status_buffer, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }

    void print_tracking_info(const std::vector<STrack> &tracks, uint32_t frame_count)
    {
        auto current_time = this->now();
        double time_since_last_log = (current_time - last_tracking_log_time_).seconds();

        // 更严格的输出条件：减少频繁输出
        if (time_since_last_log < 0.2 && tracks.size() <= 2)
        {
            return;
        }

        // 只在目标状态变化或有多目标时输出
        bool should_log = false;
        if (tracks.size() > 1)
        {
            should_log = true;
        }
        else if (tracked_target_.track_id != -1)
        {
            // 检查目标状态是否有变化
            static int last_target_id = -1;
            static bool last_hands_up = false;

            if (last_target_id != tracked_target_.track_id ||
                (tracks.size() == 1 && isHandsUp(tracks[0]) != last_hands_up))
            {
                should_log = true;
                last_target_id = tracked_target_.track_id;
                if (tracks.size() == 1)
                {
                    last_hands_up = isHandsUp(tracks[0]);
                }
            }
        }

        if (!should_log)
        {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "===== Tracking Results (Frame %u) =====", frame_count);
        RCLCPP_INFO(this->get_logger(), "Total tracks: %zu", tracks.size());

        for (const auto &track : tracks)
        {
            bool is_target_tracked = (tracked_target_.track_id == track.track_id);
            bool hands_up = isHandsUp(track);

            RCLCPP_INFO(this->get_logger(),
                        "Track ID: %d | Target: %s | HandsUp: %s | Score: %.2f",
                        track.track_id,
                        is_target_tracked ? "YES" : "NO",
                        hands_up ? "YES" : "NO",
                        track.score);
        }

        RCLCPP_INFO(this->get_logger(), "===========================");

        last_tracking_log_time_ = current_time;
    }

    void print_performance_stats(const ProcessingTimes &times, const std_msgs::msg::Header &header)
    {
        auto current_time = this->now();
        double time_since_last_log = (current_time - last_perf_log_time_).seconds();

        // 使用时间戳而不是序列号
        uint64_t timestamp_nanos = header.stamp.sec * 1000000000LL + header.stamp.nanosec;

        RCLCPP_INFO(this->get_logger(), "=== Performance Statistics (Frame %u, Time: %lu) ===",
                    total_frames_, timestamp_nanos);
        RCLCPP_INFO(this->get_logger(), "Image conversion: %.3f ms", times.image_conversion * 1000);
        RCLCPP_INFO(this->get_logger(), "Preprocessing: %.3f ms", times.preprocessing * 1000);
        RCLCPP_INFO(this->get_logger(), "Inference: %.3f ms", times.inference * 1000);
        RCLCPP_INFO(this->get_logger(), "Postprocessing: %.3f ms", times.postprocessing * 1000);
        RCLCPP_INFO(this->get_logger(), "Tracking: %.3f ms", times.tracking * 1000);
        RCLCPP_INFO(this->get_logger(), "Visualization: %.3f ms", times.visualization * 1000);
        RCLCPP_INFO(this->get_logger(), "Publishing: %.3f ms", times.publishing * 1000);
        RCLCPP_INFO(this->get_logger(), "Total processing: %.3f ms", times.total * 1000);
        RCLCPP_INFO(this->get_logger(), "Average FPS: %.2f", total_frames_ / total_processing_time_);
        RCLCPP_INFO(this->get_logger(), "=======================================");

        last_perf_log_time_ = current_time;
    }

    std::vector<Object> convert_detections_to_trackobj(
        const std::vector<YOLOv11Pose::Detection> &detections,
        const std::vector<int> &nms_indices)
    {
        std::vector<Object> trackobj;
        trackobj.reserve(nms_indices.size());

        for (int idx : nms_indices)
        {
            const auto &det = detections[idx];
            trackobj.emplace_back();
            auto &obj = trackobj.back();
            obj.classId = 0;
            obj.score = det.score;
            obj.box = cv::Rect(
                static_cast<int>(det.bbox.x),
                static_cast<int>(det.bbox.y),
                static_cast<int>(det.bbox.width),
                static_cast<int>(det.bbox.height));
            obj.keypoints = det.keypoints;
            obj.keypoints_score = det.keypoints_score;
        }
        return trackobj;
    }

private:
    // ROS相关
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr person_point_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detect_pose_pub_;
    rclcpp::Publisher<posetrack::msg::KeypointTrack>::SharedPtr keypoint_tracks_pub_;

    // 模型和跟踪器
    std::unique_ptr<YOLOv11Pose> pose_detector_;
    std::unique_ptr<BYTETracker> tracker_;
    std::unique_ptr<OSNetRDKX5Inference> osnet_;

    // 相机内参
    float fx_, fy_, cx_, cy_;

    // 参数
    float conf_threshold_;
    float kpt_conf_threshold_;
    float conf_threshold_raw_;
    float kpt_conf_threshold_raw_;
    float real_person_height_;
    int max_processing_fps_;
    float min_process_interval_;
    float reid_similarity_threshold_;
    float hands_up_duration_threshold_;
    float hands_up_cooldown_;
    int max_lost_frames_;

    // 图像处理参数
    float current_x_scale_ = 1.0f;
    float current_y_scale_ = 1.0f;
    float current_x_shift_ = 0.0f;
    float current_y_shift_ = 0.0f;
    int current_x_shift_int_ = 0;
    int current_y_shift_int_ = 0;

    // 深度图像和同步
    cv::Mat depth_image_;
    std::mutex depth_mutex_;

    // 深度滤波器
    std::unordered_map<int, DepthFilter> depth_filters_;

    // 时间控制
    rclcpp::Time last_process_time_;
    rclcpp::Time last_hands_up_detection_time_;

    // 骨架连接
    int skeleton_[38];

    // 跟踪状态管理
    std::unordered_map<int, TrackedPerson> tracked_persons_;
    TrackedTarget tracked_target_;

    // 添加性能统计变量
    int total_frames_;
    double total_processing_time_;
    std::unordered_map<std::string, double> time_stats_;

    TrackingState tracking_state_;
    rclcpp::Time tracking_start_time_;
    rclcpp::Time cooldown_start_time_;

    // 添加时间跟踪变量
    rclcpp::Time last_perf_log_time_;
    rclcpp::Time last_tracking_log_time_;
    uint32_t last_processed_seq_ = 0;

    // 添加异步处理相关变量
    std::deque<sensor_msgs::msg::Image::ConstSharedPtr> image_queue_;
    std::mutex queue_mutex_;
    std::thread processing_thread_;
    std::atomic<bool> processing_;
    std::atomic<bool> stop_processing_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Yolov11PoseNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}