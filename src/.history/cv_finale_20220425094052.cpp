
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include <app-yolo/yolo.hpp>
#include <app-ldrn/ldrn.hpp>


using namespace std;

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

static bool build_model(){

    bool success = true;
    if(!exists("yolov5s.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "yolov5s.onnx", "yolov5s.trtmodel");

    if(!exists("road-segmentation-adas.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "road-segmentation-adas.onnx", "road-segmentation-adas.trtmodel");
    
    if(!exists("ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "ldrn_kitti_resnext101_pretrained_data_grad_256x512.onnx", "ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel");

    if(!exists("new-lane.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "new-lane.onnx", "new-lane.trtmodel");
    return true;
}

static cv::Mat to_render_depth(const cv::Mat& depth){

    cv::Mat mask;
    depth.convertTo(mask, CV_8U, -5, 255);
    //mask = mask(cv::Rect(0, mask.rows * 0.18, mask.cols, mask.rows * (1 - 0.18)));
    cv::applyColorMap(mask, mask, cv::COLORMAP_PLASMA);
    return mask;
}

static void merge_images(
    const cv::Mat& image, const cv::Mat& road,
    const cv::Mat& depth,const cv::Mat& raw_image, cv::Mat& scence
){

    
    // image.copyTo(scence(cv::Rect(0, 0, image.cols, image.rows)));

    auto road_crop = road(cv::Rect(0, road.rows * 0.5, road.cols, road.rows * 0.5));

    cv::resize(road_crop, road_crop, cv::Size(image.cols*0.5, image.rows * 0.5));

    auto depth_crop = depth(cv::Rect(0, depth.rows * 0.18, depth.cols, depth.rows * (1 - 0.18)));

    cv::resize(depth_crop, depth_crop, cv::Size(image.cols*0.5, image.rows * 0.5));


    std::vector<cv::Mat> vImgs ;
	cv::Mat result;
	vImgs.push_back(road_crop);
	vImgs.push_back(depth_crop);

	vconcat(vImgs, result); //垂直方向拼接

    std::vector<cv::Mat> hImgs ;
    cv::Mat merge;
    hImgs.push_back(image);
    hImgs.push_back(result);

    cv::hconcat(hImgs, merge);

    scence = merge;

    // cv::imwrite("merge1.jpg", merge);
    // depth_crop.copyTo(scence(cv::Rect(image.cols, image.rows * 0.5, image.cols*1.5, image.rows * 1.5)));
}

static void inference(){

    //auto image = cv::imread("imgs/dashcam_00.jpg");
    auto yolov5 = Yolo::create_infer("yolov5s.trtmodel", Yolo::Type::V5, 0, 0.25, 0.45);
    auto road = Road::create_infer("road-segmentation-adas.trtmodel", 0);
    auto ldrn = Ldrn::create_infer("ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel", 0);
    auto lane = Lane::create_infer("new-lane.trtmodel", 0);

    cv::Mat image, scence;
    cv::VideoCapture cap("outclip.webm");
    float fps = cap.get(cv::CAP_PROP_FPS);
    int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    scence = cv::Mat(height * 1, width * 1.5, CV_8UC3, cv::Scalar::all(0));
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('M', 'P', 'G', '2'), fps, scence.size());
    // auto scence = cv::Mat(image.rows * 1, image.cols * 1.5, CV_8UC3, cv::Scalar::all(0));
    cv::Mat raw_image = image.clone();

    while(cap.read(image)){
        auto roadmask_fut = road->commit(image);
        auto boxes_fut = yolov5->commit(image);
        auto depth_fut = ldrn->commit(image);
        auto points_fut = lane->commit(image);
        auto roadmask = roadmask_fut.get();
        auto boxes = boxes_fut.get();
        auto depth = depth_fut.get();
        auto points = points_fut.get();
        cv::resize(depth, depth, image.size());
        cv::resize(roadmask, roadmask, image.size());

        for(auto& box : boxes){
            int cx = (box.left + box.right) * 0.5 + 0.5;
            int cy = (box.top + box.bottom) * 0.5 + 0.5;
            float distance = depth.at<float>(cy, cx) / 5;
            if(fabs(cx - (image.cols * 0.5)) <= 200 && cy >= image.rows * 0.85)
                continue;

            cv::Scalar color(0, 125, 125);
            cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

            auto name      = cocolabels[box.class_label];
            auto caption   = cv::format("%s %.2f%s", name, distance,"m");
            int text_width = cv::getTextSize(caption, 0, 1.5, 1, nullptr).width;
            cv::rectangle(image, cv::Point(box.left-3, box.bottom), cv::Point(box.left+text_width, box.bottom+50), color, -1);
            cv::putText(image, caption, cv::Point(box.left, box.bottom+40), 0, 1.5, cv::Scalar::all(0), 2, 16);
        }

        cv::Scalar colors[] = {
            cv::Scalar(255, 0, 0), 
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 0, 255),
            cv::Scalar(255, 0, 0)
        };
        for(int i = 0; i < 18; ++i){
            for(int j = 0; j < 4; ++j){
                auto& p = points[i * 4 + j];
                if(p.x > 0){
                    auto color = colors[j];
                    cv::circle(image, p, 20, color, -1, 16);
                }
            }
        }
        merge_images(image, roadmask, to_render_depth(depth),raw_image, scence);
        // cv::imwrite("test.jpg",scence);
        writer.write(scence);
        INFO("Process");
    }
    writer.release();
}

int main(){

    // 新的实现
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}



// //
// // Created by jzx
// //
// #include"cv_all_ros.hpp"
// using namespace cv;

// void CvAll::init() {
    
//     ROS_INFO("<< cv all go!");
//     //camera_array/camera/image_raw/compressed
//     img_sub = nh_.subscribe<sensor_msgs::CompressedImage>("camera/color/image_raw/compressed", 1,
//                                                           &CvAll::imgCallback, this);
//     nh_.param<bool>("is_debug", debug_, true);

//     //segmentation-model
//     ROS_INFO("<< add segmentation model!");
//     std::string segmentaion_model_file = "/home/jzx/IMPORTANT_MODELS/unet.engine";
//     cs = std::make_shared<CvSegmentation>();
//     cs->getEngine(segmentaion_model_file);
//     //classification-model


//     //detection-model
//     ROS_INFO("<< add detection model!");
//     std::string detection_model_file = "/home/jzx/IMPORTANT_MODELS/yolov5.engine";
//     cd = std::make_shared<CvDetection>();
//     cd->getEngine(detection_model_file);
// }



// void CvAll::imgCallback(const sensor_msgs::CompressedImage::ConstPtr &image_msg) {
//     try {
//         if(image_msg->header.seq%1==0){
//             cv::Mat image = cv::imdecode(cv::Mat(image_msg->data), 1);//convert compressed image data to cv::Mat

//             cv::Mat clone_image = image.clone();

//             auto detection_img = cd->inference(image);
//             auto segmentation_img = cs->inference(clone_image);
//             cv::Mat show_img;
//             cv::hconcat(detection_img, segmentation_img,show_img);
            
//             cv::resize(show_img, show_img,cv::Size(1200, 400));
//             cv::imshow("detection", show_img);
//             cv::waitKey(1);
//         }
//     }
//     catch (cv_bridge::Exception &e) {
//         std::cout<<"could not "<<std::endl;
//     }
// }

