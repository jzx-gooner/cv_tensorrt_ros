#ifndef CV_ALL_ROS_H
#define CV_ALL_ROS_H
// Created by jzx

//ros+opencv
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "sensor_msgs/CompressedImage.h"
#include <image_transport/image_transport.h>

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
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>

#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include <app-yolo/yolo.hpp>
#include <app-ldrn/ldrn.hpp>
#include <app-bisenet/bisenet.hpp>

using namespace TRT;

class CvAll {

public:

    //ros
    ros::NodeHandle nh_;

    void init();

    void imgCallback(const sensor_msgs::CompressedImage::ConstPtr &image_msg);

    void inference(cv::Mat &image);

    void sendMsgs(sensor_msgs::ImagePtr msg);

private:
    ros::Subscriber img_sub;
    ros::Publisher cvInfo_pub;
    bool debug_ = false;

    std::shared_ptr<Yolo::Infer> detection_infer_;
    std::shared_ptr<Ldrn::Infer> depth_infer_;
    std::shared_ptr<Bisenet::Infer> segmentation_infer_;
    // std::shared_ptr<Infer> classification_infer_;

};

#endif