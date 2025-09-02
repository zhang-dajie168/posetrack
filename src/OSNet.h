#ifndef OSNET_H
#define OSNET_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

// 使用与 yolov11_pose 相同的头文件
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/hb_sys.h"

class OSNetRDKX5Inference {
public:
    /**
     * @brief 构造函数
     * @param bin_model_path 模型文件路径
     * @param input_size 输入图像尺寸，默认 64x128
     */
    OSNetRDKX5Inference(const std::string& bin_model_path, 
                       const cv::Size& input_size = cv::Size(64, 128));
    
    /**
     * @brief 析构函数
     */
    ~OSNetRDKX5Inference();
    
    /**
     * @brief 初始化模型
     * @return 是否初始化成功
     */
    bool init_model();
    
    /**
     * @brief 提取图像特征
     * @param image 输入图像
     * @return 特征向量
     */
    std::vector<float> extract_features(const cv::Mat& image);
    
    /**
     * @brief 批量提取图像特征
     * @param images 输入图像列表
     * @return 特征向量列表
     */
    std::vector<std::vector<float>> extract_features_batch(const std::vector<cv::Mat>& images);
    
    /**
     * @brief 计算两个特征向量的相似度
     * @param features1 第一个特征向量
     * @param features2 第二个特征向量
     * @return 相似度得分 (0.0 - 1.0)
     */
    float compute_similarity(const std::vector<float>& features1, 
                            const std::vector<float>& features2);
    
    /**
     * @brief 获取性能统计信息
     * @param avg_preprocess_ms 平均预处理时间(毫秒)
     * @param avg_inference_ms 平均推理时间(毫秒)
     * @param total_count 总处理次数
     */
    void get_performance_stats(double& avg_preprocess_ms, 
                              double& avg_inference_ms,
                              int& total_count);

private:
    /**
     * @brief 检查文件是否存在
     * @param path 文件路径
     * @return 文件是否存在
     */
    bool fileExists(const std::string& path);
    
    /**
     * @brief BGR图像转NV12格式
     * @param bgr_image 输入的BGR图像
     * @param nv12_buffer 输出的NV12缓冲区
     */
    void bgr2nv12_simple_fast(const cv::Mat& bgr_image, std::vector<uint8_t>& nv12_buffer);
    
    /**
     * @brief 预处理图像
     * @param image 输入图像
     * @param nv12_buffer 输出的NV12缓冲区
     * @return 预处理耗时(微秒)
     */
    int64_t preprocess_image(const cv::Mat& image, std::vector<uint8_t>& nv12_buffer);
    
    /**
     * @brief 后处理输出，提取特征向量
     * @param output_tensor 输出张量
     * @return 特征向量
     */
    std::vector<float> postprocess(hbDNNTensor* output_tensor);

    std::string model_path_;       // 模型文件路径
    cv::Size input_size_;          // 输入图像尺寸
    int width_;                    // 图像宽度
    int height_;                   // 图像高度
    
    // 模型上下文
    hbPackedDNNHandle_t packed_dnn_handle_ = nullptr;
    hbDNNHandle_t dnn_handle_ = nullptr;
    hbDNNTensorProperties input_properties_;
    int output_count_ = 0;
    
    // 输入输出张量缓存
    hbDNNTensor input_tensor_;
    hbDNNTensor* output_tensors_ = nullptr;
    
    // 性能统计
    int64_t total_preprocess_time_ = 0;  // 总预处理时间(微秒)
    int64_t total_inference_time_ = 0;   // 总推理时间(微秒)
    int process_count_ = 0;              // 处理次数
};

#endif // OSNET_H