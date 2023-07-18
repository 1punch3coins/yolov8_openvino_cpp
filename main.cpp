#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "algorithms/yolov8.h"

int main(int argc, char* argv[]) {
    // cv::Mat img = cv::imread("./resource/inputs/dog.jpg");
    // cv::Mat resized_img;
    // cv::resize(img, resized_img, cv::Size(640, 640));
    // // 1. 
    // ov::Core core;
    // // 2.
    // std::shared_ptr<ov::Model> model = core.read_model("./resource/models/yolov8s_post_fp32.xml");
    // std::vector<ov::Output<ov::Node>> input_nodes = model->inputs();
    // std::vector<ov::Output<ov::Node>> output_nodes = model->outputs();
    // std::string input_name = input_nodes[0].get_any_name();
    // std::string output_name = output_nodes[0].get_any_name();
    // int32_t input_node_index = input_nodes[0].get_index();
    // int32_t output_node_index = output_nodes[0].get_index();
    // ov::Shape input_shape = input_nodes[0].get_shape();
    // ov::Shape output_shape = output_nodes[0].get_shape();
    // // ov::element::Type input_type = input_nodes[0].get_element_type();
    // // ov::element::Type output_type = output_nodes[0].get_element_type();
    // // 3.
    // ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    // ov::preprocess::InputInfo& input_info = ppp.input();
    // ov::preprocess::OutputInfo& output_info = ppp.output();
    // input_info.tensor().set_element_type(ov::element::u8);
    // input_info.tensor().set_layout("NCHW").set_color_format(ov::preprocess::ColorFormat::BGR);
    // input_info.preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    // input_info.model().set_layout("NCHW");
    // output_info.tensor().set_element_type(ov::element::f32);
    // model = ppp.build();
    // const int32_t batch_size = 1;
    // ov::set_batch(model, batch_size);
    // ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    // // 4.
    // float* input_data = (float*)resized_img.data;
    // uint8_t* cc = (uint8_t*)resized_img.data;
    // // ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
    // auto aa = compiled_model.input().get_element_type();
    // auto bb = compiled_model.input().get_shape();
    // ov::Tensor input_tensor = ov::Tensor(ov::element::u8, input_shape, input_data);
    // // 5.
    // ov::InferRequest infer_request = compiled_model.create_infer_request();
    // infer_request.set_input_tensor(input_tensor);
    // infer_request.infer();
    // const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    // float* output = output_tensor.data<float>();
    Yolov8 yolo;
    if (yolo.Initialize("./resource/models/yolov8s_post_fp32.xml") != 1) {
        std::cout << "yolo initialization uncompleted" << std::endl;
        return 0;
    }

    cv::Mat original_img = cv::imread("./resource/inputs/test1.png");
    Yolov8::Result det_res;
    if (yolo.Process(original_img, det_res) != 1) {
        std::cout << "yolo forward uncompleted" << std::endl;
        return 0;
    }
    for (const auto& box: det_res.bbox_list) {
        cv::putText(original_img, box.cls_name, cv::Point(box.x, box.y - 6), 0, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(original_img, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("./resource/outputs/output1.png", original_img);

    return 0;
}