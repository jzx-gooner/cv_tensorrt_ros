//
// Created by jzx
//
#include <stdio.h>
#include <string.h>
#include "cv_all_ros.hpp"

int main(int argc, char **argv) {
    ros::init(argc, argv, "cv_all");
    CvAll ca;
    ca.init();
    // ROS_INFO("<< cv all go!");
    // ros::spin();
    return 0;
}