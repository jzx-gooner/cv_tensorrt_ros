#ifndef CV_ALL_ROS_H
#define CV_ALL_ROS_H

//
// Created by jzx
//
//ros+opencv
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <cmath>
#include <unistd.h>
#include "sensor_msgs/CompressedImage.h"
#include <image_transport/image_transport.h>

//sgementation
#include "cv_segmentation.hpp"
#include "cv_detection.hpp"




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

using namespace TRT;

class CvAll {

public:

    //ros
    ros::NodeHandle nh_;

    void init();

    void imgCallback(const sensor_msgs::CompressedImage::ConstPtr &image_msg);

    void sendMsgs(sensor_msgs::ImagePtr msg);

private:
    ros::Subscriber img_sub;
    ros::Publisher cvInfo_pub;
    bool debug_ = false;

    std::shared_ptr<CvSegmentation> cs;
    std::shared_ptr<CvDetection> cd;
    // std::shared_ptr<CvSegmentation> cs;
    // std::shared_ptr<Infer> classification_infer_;
    // std::shared_ptr<Infer> detection_infer_;


};

#endif