#include "cv_all_ros.hpp"

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
    return access(path.c_str(), R_OK) == 0;
}


// static cv::Mat to_render_depth(const cv::Mat& depth){

//     cv::Mat mask;
//     depth.convertTo(mask, CV_8U, -5, 255);
//     //mask = mask(cv::Rect(0, mask.rows * 0.18, mask.cols, mask.rows * (1 - 0.18)));
//     cv::applyColorMap(mask, mask, cv::COLORMAP_PLASMA);
//     return mask;
// }

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

static bool build_model(){

    bool success = true;
    if(!exists("/home/jzx/usv_models/yolov5s.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "/home/jzx/usv_models/yolov5s.onnx", "/home/jzx/usv_models/yolov5s.engine");

    if(!exists("/home/jzx/usv_models/depth_estimation.onnx"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "/home/jzx/usv_models/depth_estimation.onnx", "/home/jzx/usv_models/depth_estimation.engine");

    return true;
}


void CvAll::init() {
    
    ROS_INFO("<< cv all go!");
    //camera_array/camera/image_raw/compressed
    img_sub = nh_.subscribe<sensor_msgs::CompressedImage>("camera/color/image_raw/compressed", 1,
                                                          &CvAll::imgCallback, this);
    nh_.param<bool>("is_debug", debug_, true);

    //如果没有模型，则编译模型
    build_model();
    //加载模型
    detection_infer_ = Yolo::create_infer("yolov5s.trtmodel", Yolo::Type::V5, 0, 0.25, 0.45);
    depth_infer_ = Ldrn::create_infer("ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel", 0);

}


void CvAll::imgCallback(const sensor_msgs::CompressedImage::ConstPtr &image_msg) {
    try {
        if(image_msg->header.seq%2==0){
            cv::Mat image = cv::imdecode(cv::Mat(image_msg->data), 1);//convert compressed image data to cv::Mat
            inference(image);
        }
    }
    catch (cv_bridge::Exception &e) {
        std::cout<<"could not "<<std::endl;
    }
}

void CvAll::inference(cv::Mat &image){

        auto boxes_fut = detection_infer_->commit(image);
        auto depth_fut = depth_infer_->commit(image);
        auto boxes = boxes_fut.get();
        auto depth = depth_fut.get();
        cv::resize(depth, depth, image.size());

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
        INFO("Process");
}
