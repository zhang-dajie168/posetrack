#include "OSNet.h"
#include <fstream>
#include <chrono>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// 自定义 clamp 函数，兼容 C++11
template<typename T>
T clamp_value(T value, T min_val, T max_val) {
    return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}

// 错误检查宏
#define OSNET_CHECK_SUCCESS(value, errmsg)                      \
    do                                                          \
    {                                                           \
        auto ret_code = value;                                  \
        if (ret_code != 0)                                      \
        {                                                       \
            throw std::runtime_error(errmsg + std::string(", error code: ") + std::to_string(ret_code)); \
        }                                                       \
    } while (0);

// 构造函数
OSNetRDKX5Inference::OSNetRDKX5Inference(const std::string &bin_model_path,
                                         const cv::Size &input_size)
    : model_path_(bin_model_path),
      input_size_(input_size),
      width_(input_size.width),
      height_(input_size.height)
{
    // 初始化输入张量
    memset(&input_tensor_, 0, sizeof(input_tensor_));
    
    // 检查模型文件是否存在
    if (!fileExists(bin_model_path))
    {
        throw std::runtime_error("模型文件不存在: " + bin_model_path);
    }
}

// 析构函数
OSNetRDKX5Inference::~OSNetRDKX5Inference()
{
    // 释放输入张量内存
    if (input_tensor_.sysMem[0].virAddr)
    {
        hbSysFreeMem(&input_tensor_.sysMem[0]);
        input_tensor_.sysMem[0].virAddr = nullptr;
    }

    // 释放输出张量内存
    if (output_tensors_)
    {
        for (int i = 0; i < output_count_; i++)
        {
            if (output_tensors_[i].sysMem[0].virAddr)
            {
                hbSysFreeMem(&output_tensors_[i].sysMem[0]);
                output_tensors_[i].sysMem[0].virAddr = nullptr;
            }
        }
        delete[] output_tensors_;
        output_tensors_ = nullptr;
    }

    // 释放模型句柄
    if (dnn_handle_)
    {
        hbDNNRelease(dnn_handle_);
        dnn_handle_ = nullptr;
    }

    // 释放打包模型句柄
    if (packed_dnn_handle_)
    {
        hbDNNRelease(packed_dnn_handle_);
        packed_dnn_handle_ = nullptr;
    }
}

// 初始化模型
bool OSNetRDKX5Inference::init_model()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    try
    {
        // 1. 加载bin模型
        const char *model_file_name = model_path_.c_str();
        OSNET_CHECK_SUCCESS(
            hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_file_name, 1),
            "hbDNNInitializeFromFiles failed");

        // 2. 获取模型句柄
        const char **model_name_list;
        int model_count = 0;
        OSNET_CHECK_SUCCESS(
            hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle_),
            "hbDNNGetModelNameList failed");

        const char *model_name = model_name_list[0];
        OSNET_CHECK_SUCCESS(
            hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name),
            "hbDNNGetModelHandle failed");

        // 3. 获取输入属性
        OSNET_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_properties_, dnn_handle_, 0),
            "hbDNNGetInputTensorProperties failed");

        // 4. 获取输出数量
        OSNET_CHECK_SUCCESS(
            hbDNNGetOutputCount(&output_count_, dnn_handle_),
            "hbDNNGetOutputCount failed");

        std::cout << "RDK X5模型加载成功" << std::endl;
        std::cout << "输入形状: ["
                  << input_properties_.validShape.dimensionSize[0] << ", "
                  << input_properties_.validShape.dimensionSize[1] << ", "
                  << input_properties_.validShape.dimensionSize[2] << ", "
                  << input_properties_.validShape.dimensionSize[3] << "]" << std::endl;

        // 获取输出信息
        for (int i = 0; i < output_count_; i++)
        {
            hbDNNTensorProperties output_properties;
            if (hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, i) == 0)
            {
                std::cout << "输出" << i << "形状: [";
                for (int j = 0; j < output_properties.validShape.numDimensions; j++)
                {
                    std::cout << output_properties.validShape.dimensionSize[j] << " ";
                }
                std::cout << "]" << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        // 清理资源
        if (dnn_handle_)
        {
            hbDNNRelease(dnn_handle_);
            dnn_handle_ = nullptr;
        }
        if (packed_dnn_handle_)
        {
            hbDNNRelease(packed_dnn_handle_);
            packed_dnn_handle_ = nullptr;
        }
        throw std::runtime_error("模型加载失败: " + std::string(e.what()));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "模型加载耗时: " << duration.count() << "ms" << std::endl;

    return true;
}

// BGR 转 NV12
void OSNetRDKX5Inference::bgr2nv12_simple_fast(const cv::Mat &bgr_image, std::vector<uint8_t>& nv12_buffer)
{
    cv::Mat resized;
    cv::resize(bgr_image, resized, input_size_, 0, 0, cv::INTER_LINEAR);

    // 直接操作 NV12 缓冲区
    uint8_t* y_plane = nv12_buffer.data();
    uint8_t* uv_plane = nv12_buffer.data() + height_ * width_;
    
    // 转换为 YUV 并直接填充到 NV12 缓冲区
    for (int i = 0; i < height_; ++i)
    {
        for (int j = 0; j < width_; ++j)
        {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(i, j);
            uint8_t b = pixel[0], g = pixel[1], r = pixel[2];
            
            // Y 分量计算公式
            int y_val = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            y_plane[i * width_ + j] = clamp_value(y_val, 0, 255);
            
            // 每 2x2 块计算一次 UV
            if (i % 2 == 0 && j % 2 == 0)
            {
                int u_idx = (i / 2) * (width_ / 2) + (j / 2);
                
                // U 分量
                int u_val = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                uv_plane[2 * u_idx] = clamp_value(u_val, 0, 255);
                
                // V 分量
                int v_val = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                uv_plane[2 * u_idx + 1] = clamp_value(v_val, 0, 255);
            }
        }
    }
}

// 预处理图像
int64_t OSNetRDKX5Inference::preprocess_image(const cv::Mat &image, std::vector<uint8_t>& nv12_buffer)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // 确保缓冲区大小正确
    nv12_buffer.resize(height_ * 3 / 2 * width_);
    bgr2nv12_simple_fast(image, nv12_buffer);

    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
}

// 提取特征
std::vector<float> OSNetRDKX5Inference::extract_features(const cv::Mat &image)
{
    try
    {
        // 预处理
        std::vector<uint8_t> nv12_buffer;
        int64_t preprocess_time = preprocess_image(image, nv12_buffer);
        total_preprocess_time_ += preprocess_time;

        // 准备输入张量（重用已分配的内存）
        if (input_tensor_.sysMem[0].virAddr == nullptr) {
            input_tensor_.properties = input_properties_;
            OSNET_CHECK_SUCCESS(
                hbSysAllocCachedMem(&input_tensor_.sysMem[0], nv12_buffer.size()),
                "hbSysAllocCachedMem failed");
        }
        
        memcpy(input_tensor_.sysMem[0].virAddr, nv12_buffer.data(), nv12_buffer.size());
        hbSysFlushMem(&input_tensor_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

        // 准备输出张量（重用已分配的内存）
        if (output_tensors_ == nullptr) {
            output_tensors_ = new hbDNNTensor[output_count_];
            for (int i = 0; i < output_count_; i++)
            {
                hbDNNTensorProperties output_properties;
                OSNET_CHECK_SUCCESS(
                    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, i),
                    "hbDNNGetOutputTensorProperties failed");
                output_tensors_[i].properties = output_properties;
                OSNET_CHECK_SUCCESS(
                    hbSysAllocCachedMem(&output_tensors_[i].sysMem[0], output_properties.alignedByteSize),
                    "hbSysAllocCachedMem failed");
            }
        }

        // 推理
        auto start_time = std::chrono::high_resolution_clock::now();
        hbDNNTaskHandle_t task_handle = nullptr;
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

        OSNET_CHECK_SUCCESS(
            hbDNNInfer(&task_handle, &output_tensors_, &input_tensor_, dnn_handle_, &infer_ctrl_param),
            "hbDNNInfer failed");

        OSNET_CHECK_SUCCESS(
            hbDNNWaitTaskDone(task_handle, 0),
            "hbDNNWaitTaskDone failed");

        if (task_handle)
        {
            hbDNNReleaseTask(task_handle);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        total_inference_time_ += duration.count();
        process_count_++;

        // 后处理
        auto features = postprocess(output_tensors_);

        return features;
    }
    catch (const std::exception &e)
    {
        std::cerr << "特征提取错误: " << e.what() << std::endl;
        throw;
    }
}

// 后处理输出
std::vector<float> OSNetRDKX5Inference::postprocess(hbDNNTensor *output_tensor)
{
    // 假设第一个输出是特征向量
    hbSysFlushMem(&(output_tensor[0].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    float *feature_data = reinterpret_cast<float *>(output_tensor[0].sysMem[0].virAddr);
    size_t feature_size = 1;
    for (int i = 0; i < output_tensor[0].properties.validShape.numDimensions; i++)
    {
        feature_size *= output_tensor[0].properties.validShape.dimensionSize[i];
    }

    return std::vector<float>(feature_data, feature_data + feature_size);
}

// 批量提取特征
std::vector<std::vector<float>> OSNetRDKX5Inference::extract_features_batch(const std::vector<cv::Mat> &images)
{
    std::vector<std::vector<float>> all_features;
    all_features.reserve(images.size());

    // 预热
    if (process_count_ == 0 && !images.empty()) {
        extract_features(images[0]);
    }

    for (const auto &image : images)
    {
        try
        {
            auto features = extract_features(image);
            all_features.push_back(features);
        }
        catch (const std::exception &e)
        {
            std::cerr << "批量处理错误: " << e.what() << std::endl;
            // 添加空特征向量但保持相同维度
            std::vector<float> empty_features(512, 0.0f); // 假设特征维度为512
            all_features.push_back(empty_features);
        }
    }

    return all_features;
}

// 计算相似度
float OSNetRDKX5Inference::compute_similarity(const std::vector<float> &features1,
                                              const std::vector<float> &features2)
{
    if (features1.empty() || features2.empty() ||
        features1.size() != features2.size())
    {
        return 0.0f;
    }

    // 计算 L2 范数
    float norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < features1.size(); ++i)
    {
        norm1 += features1[i] * features1[i];
        norm2 += features2[i] * features2[i];
    }
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 <= 0 || norm2 <= 0)
    {
        return 0.0f;
    }

    // 归一化并计算点积
    float similarity = 0.0f;
    for (size_t i = 0; i < features1.size(); ++i)
    {
        similarity += (features1[i] / norm1) * (features2[i] / norm2);
    }

    return similarity;
}

// 获取性能统计
void OSNetRDKX5Inference::get_performance_stats(double &avg_preprocess_ms,
                                                double &avg_inference_ms,
                                                int &total_count)
{
    if (process_count_ == 0)
    {
        avg_preprocess_ms = 0;
        avg_inference_ms = 0;
        total_count = 0;
        return;
    }

    avg_preprocess_ms = (total_preprocess_time_ / 1000.0) / process_count_;
    avg_inference_ms = (total_inference_time_ / 1000.0) / process_count_;
    total_count = process_count_;
    
    // 内存使用统计
    size_t total_memory = 0;
    if (input_tensor_.sysMem[0].virAddr) {
        total_memory += input_tensor_.properties.alignedByteSize;
    }
    if (output_tensors_) {
        for (int i = 0; i < output_count_; i++) {
            total_memory += output_tensors_[i].properties.alignedByteSize;
        }
    }
    
    std::cout << "内存使用: " << total_memory / 1024.0 << " KB" << std::endl;
}

// 检查文件是否存在
bool OSNetRDKX5Inference::fileExists(const std::string &path)
{
    std::ifstream file(path);
    return file.good();
}