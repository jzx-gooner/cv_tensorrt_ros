//
// Created by jzx
//
#include"cv_all_ros.hpp"
using namespace cv;

void CvAll::init() {
    
    ROS_INFO("<< cv all go!");
    //camera_array/camera/image_raw/compressed
    img_sub = nh_.subscribe<sensor_msgs::CompressedImage>("camera/color/image_raw/compressed", 1,
                                                          &CvAll::imgCallback, this);
    nh_.param<bool>("is_debug", debug_, true);

    //segmentation-model
    ROS_INFO("<< add segmentation model!");
    std::string segmentaion_model_file = "/home/jzx/IMPORTANT_MODELS/unet.engine";
    cs = std::make_shared<CvSegmentation>();
    cs->getEngine(segmentaion_model_file);
    //classification-model


    //detection-model
    ROS_INFO("<< add detection model!");
    std::string detection_model_file = "/home/jzx/IMPORTANT_MODELS/yolov5.engine";
    cd = std::make_shared<CvDetection>();
    cd->getEngine(detection_model_file);
}



void CvAll::imgCallback(const sensor_msgs::CompressedImage::ConstPtr &image_msg) {
    try {
        if(image_msg->header.seq%1==0){
            cv::Mat image = cv::imdecode(cv::Mat(image_msg->data), 1);//convert compressed image data to cv::Mat

            cv::Mat clone_image = image.clone();

            auto detection_img = cd->inference(image);
            auto segmentation_img = cs->inference(clone_image);
            cv::Mat show_img;
            cv::hconcat(detection_img, segmentation_img,show_img);
            
            cv::resize(show_img, show_img,cv::Size(1200, 400));
            cv::imshow("detection", show_img);
            cv::waitKey(1);
        }
    }
    catch (cv_bridge::Exception &e) {
        std::cout<<"could not "<<std::endl;
    }
}

