#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <ie_core.hpp>

#include "openvino_bridge.h"
#define INTEGRATE_NORMALIZATION true

OpenvinoBridge* OpenvinoBridge::Create(const std::string& model_pwd, NetworkMeta* p_meta) {
    OpenvinoBridge* p = new OpenvinoBridge();
    p->model_pwd_ = model_pwd;
    p->network_meta_.reset(p_meta);
    return p;
}

int32_t OpenvinoBridge::Initialize() {
    // 0. Check device's support model percision settings
    // InferenceEngine::Core core;
    // auto cpuOptimizationCapabilities = core.GetMetric("CPU", METRIC_KEY(OPTIMIZATION_CAPABILITIES)).as<std::vector<std::string>>();
    // 1.1 Readin model's input and output metadata
    model_ = core_.read_model("./resource/models/yolov8s_post_fp32.xml");
    int32_t input_meta_list_index = 0;
    int32_t output_meta_list_index = 0;
    std::map<int32_t, int32_t> input_indexs_gpu2meta;
    std::map<int32_t, int32_t> output_indexs_gpu2meta;
    for (auto& input_tensor_meta : network_meta_->input_tensor_meta_list) {
        const auto& input_node = model_->input(input_tensor_meta.tensor_name);
        int32_t gpu_input_index = input_node.get_index();
        input_indexs_gpu2meta.insert({gpu_input_index, input_meta_list_index++});
        ov::Shape input_shape = input_node.get_shape();
        if (network_meta_->input_nchw) {
            input_tensor_meta.net_in_c = input_shape[1];
            input_tensor_meta.net_in_h = input_shape[2];
            input_tensor_meta.net_in_w = input_shape[3];
        } else {
            input_tensor_meta.net_in_h = input_shape[1];
            input_tensor_meta.net_in_w = input_shape[2];
            input_tensor_meta.net_in_c = input_shape[3];
        }
        input_tensor_meta.input_scale = 1.0;
        input_tensor_meta.net_in_elements = network_meta_->batch_size * input_tensor_meta.net_in_c * input_tensor_meta.net_in_h * input_tensor_meta.net_in_w;
    }
    for (auto& output_tensor_meta : network_meta_->output_tensor_meta_list) {
        const auto& output_node = model_->output(output_tensor_meta.tensor_name);
        int32_t gpu_output_index = output_node.get_index();
        output_indexs_gpu2meta.insert({gpu_output_index, output_meta_list_index++});
        ov::Shape output_shape = output_node.get_shape();
        if (output_shape.size() == 3) {
            if (output_tensor_meta.output_nlc) {
                output_tensor_meta.net_out_l = output_shape[1];
                output_tensor_meta.net_out_c = output_shape[2];
            } else {
                output_tensor_meta.net_out_c = output_shape[1];
                output_tensor_meta.net_out_l = output_shape[2];
            }
        }
        if (output_shape.size() == 4) {
            if (output_tensor_meta.output_nlc) {
                output_tensor_meta.net_out_h = output_shape[1];
                output_tensor_meta.net_out_w = output_shape[2];
                output_tensor_meta.net_out_c = output_shape[3];
                output_tensor_meta.net_out_l = output_shape[1] * output_shape[2];
            } else {
                output_tensor_meta.net_out_c = output_shape[1];
                output_tensor_meta.net_out_h = output_shape[2];
                output_tensor_meta.net_out_w = output_shape[3];
                output_tensor_meta.net_out_l = output_shape[2] * output_shape[3];
            }
        }
        output_tensor_meta.net_out_elements = network_meta_->batch_size * output_tensor_meta.net_out_c * output_tensor_meta.net_out_l;
    }
    network_meta_->input_tensor_num = network_meta_->input_tensor_meta_list.size();
    network_meta_->output_tensor_num = network_meta_->output_tensor_meta_list.size();
    
    // 1.2 Rearrange input meta and output meta to match model's input and output order 
    for (int32_t i = 0; i < network_meta_->input_tensor_num; i++) {
        int32_t input_index = input_indexs_gpu2meta.find(i)->second;
        network_meta_->input_tensor_meta_list.push_back(network_meta_->input_tensor_meta_list[input_index]);
        network_meta_->input_name2index.insert({network_meta_->input_tensor_meta_list[input_index].tensor_name, i});
    }
    for (int32_t i = 0; i < network_meta_->output_tensor_num; i++) {
        int32_t output_index = output_indexs_gpu2meta.find(i + network_meta_->input_tensor_num)->second;
        network_meta_->output_tensor_meta_list.push_back(network_meta_->output_tensor_meta_list[output_index]);
        network_meta_->output_name2index.insert({network_meta_->output_tensor_meta_list[output_index].tensor_name, i});
    }
    network_meta_->input_tensor_meta_list.assign(network_meta_->input_tensor_meta_list.begin()+network_meta_->input_tensor_num, network_meta_->input_tensor_meta_list.end());
    network_meta_->output_tensor_meta_list.assign(network_meta_->output_tensor_meta_list.begin()+network_meta_->output_tensor_num, network_meta_->output_tensor_meta_list.end());
    // TO DO CHECK

    // 2. Deseralize openvino model & integerate normalization into model(optional)
#if INTEGRATE_NORMALIZATION 
    auto ppp = ov::preprocess::PrePostProcessor(model_);
    for (const auto& input_tensor_meta : network_meta_->input_tensor_meta_list) {
        ov::preprocess::InputInfo& input_info = ppp.input(input_tensor_meta.tensor_name);
        input_info.tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
        input_info.preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
        input_info.model().set_layout("NCHW");
    }
    for (const auto& output_tensor_meta : network_meta_->output_tensor_meta_list) {
        ov::preprocess::OutputInfo& output_info = ppp.output(output_tensor_meta.tensor_name);
        output_info.tensor().set_element_type(ov::element::f32);
    }
    model_ = ppp.build();
#endif
    const int32_t batch_size = 1;
    ov::set_batch(model_, batch_size);
    compiled_model_ = core_.compile_model(model_, "CPU");
    infer_request_ = compiled_model_.create_infer_request();
    for (int32_t i = 0; i < network_meta_->input_tensor_num; i++) {
        input_shapes_.push_back(compiled_model_.input(i).get_shape());
    }
    for (int32_t i = 0; i < network_meta_->output_tensor_num; i++) {
        output_shapes_.push_back(compiled_model_.output(i).get_shape());
    }

    input_ptrs_.resize(network_meta_->input_tensor_num);
    output_ptrs_.reserve(network_meta_->output_tensor_num);

    return 1;
}

int32_t OpenvinoBridge::SetCropAttr(const int32_t src_w, int32_t src_h, const int32_t dst_w, const int32_t dst_h, const kCropStyle style) {
    if (style == kCropStyle::CropAll_Coverup) {    // retain
        src_crop_.x = 0;
        src_crop_.y = 0;
        src_crop_.width = src_w;
        src_crop_.height = src_h;
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        return 1;
    }
    if (style == kCropStyle::CropLower_Coverup_1) {    // a very distinct one, shed 0.4 top part of src, disgard of ratio
        src_crop_.x = 0;
        src_crop_.y = src_h * 0.4;
        src_crop_.width = src_w;
        src_crop_.height = src_h - src_crop_.y;
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        return 1;
    }
    if (style == kCropStyle::CropLower_Coverup_0) {    // shed top part of src, retain width and make the crop ratio equals to model's input's
        float src_ratio = 1.0 * src_h / src_w;
        float dst_ratio = 1.0 * dst_h / dst_w;
        if (src_ratio > dst_ratio) {
            src_crop_.width = src_w;
            src_crop_.height = static_cast<int32_t>(src_w * dst_ratio);
            src_crop_.x = 0;
            src_crop_.y = src_h - src_crop_.height;
        } else {
            src_crop_.width = src_w;
            src_crop_.height = src_h;
            src_crop_.x = 0;
            src_crop_.y = 0;
        }
        dst_crop_.x = 0;
        dst_crop_.y = 0;
        dst_crop_.width = dst_w;
        dst_crop_.height = dst_h;
        return 1;
    }
    if (style == kCropStyle::CropAll_Embedd) {    // embedd src into dst's center, src's ratio not changed
        src_crop_.x = 0;
        src_crop_.y = 0;
        src_crop_.width = src_w;
        src_crop_.height = src_h;
        float src_ratio = 1.0 * src_w / src_h;
        float dst_ratio = 1.0 * dst_w / dst_h;
        if (src_ratio > dst_ratio) {
            // Use dst's width as base
            dst_crop_.width = dst_w;
            // dst_crop_.height = dst_h * dst_ratio / src_ratio;
            // dst_crop_.height = src_h * (dst_w / src_w);
            dst_crop_.height = static_cast<int32_t>(dst_w / src_ratio);
            dst_crop_.x = 0;
            dst_crop_.y = (dst_h - dst_crop_.height) / 2;
        } else {
            // Use dst's height as base
            dst_crop_.height = dst_h;
            dst_crop_.width = static_cast<int32_t>(dst_h / src_ratio);
            dst_crop_.x = (dst_w - dst_crop_.width) / 2;
            dst_crop_.y = 0;
        }
    }
    return 1;
}

int32_t OpenvinoBridge::PreProcess(cv::Mat& original_img, const std::string& input_name, const kCropStyle& style) {
    int32_t input_index = network_meta_->input_name2index.find(input_name)->second;
    auto tensor_meta = network_meta_->input_tensor_meta_list[input_index];
    int32_t input_h = tensor_meta.net_in_h;
    int32_t input_w = tensor_meta.net_in_w;
    int32_t input_c = tensor_meta.net_in_c;
    SetCropAttr(original_img.cols, original_img.rows, input_w, input_h, style);
    const auto& t0 = std::chrono::steady_clock::now();
    cv::Mat sample = cv::Mat::zeros(input_w, input_h, CV_8UC3);
    cv::Mat resized_mat = sample(dst_crop_);
    cv::resize(original_img, resized_mat, resized_mat.size(), 0, 0, cv::INTER_NEAREST); // Why must assign fx and fy to enable deep copy?
    input_ptrs_[input_index] = (uint8_t*)sample.data;
    // if (network_meta_->input_rgb) {
    //     cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);
    // }
    // const auto& t1 = std::chrono::steady_clock::now();
    // uint8_t* src = (uint8_t*)sample.data;
    // PermuateAndNormalize((float*)input_ptrs_[input_index], src, input_h, input_w, input_c);
    // const auto& t2 = std::chrono::steady_clock::now();
    // std::cout << "---" << 1.0 * (t1 - t0).count() * 1e-6 << std::endl;
    // std::cout << "---" << 1.0 * (t2 - t1).count() * 1e-6 << std::endl;
    return 1;
}

int32_t OpenvinoBridge::Inference() {
    for (int32_t i = 0; i < network_meta_->input_tensor_num; i++) {
        ov::Tensor input_tensor = ov::Tensor(ov::element::u8, input_shapes_[i], input_ptrs_[i]);
        infer_request_.set_input_tensor(i, input_tensor);
    }
    infer_request_.infer();
    for (int32_t i = 0; i < network_meta_->output_tensor_num; i++) {
        const ov::Tensor& output_tensor = infer_request_.get_output_tensor(i);
        output_ptrs_.push_back(output_tensor.data<float>());
    }
}

int32_t OpenvinoBridge::Finalize() {
    return 1;
}