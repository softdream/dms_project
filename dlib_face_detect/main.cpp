#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <dlib/opencv.h>
#include "opencv2/opencv.hpp"
#include <vector>
#include <ctime>

#include <chrono>
 
//using namespace dlib;
using namespace std;
//using namespace cv;
 
void line_one_face_detections(cv::Mat img, std::vector<dlib::full_object_detection> fs)
{
    int i, j;
    for(j=0; j<fs.size(); j++)
    {
        cv::Point p1, p2;
        for(i = 0; i<67; i++)
        {
            // 下巴到脸颊 0 ~ 16
            //左边眉毛 17 ~ 21
            //右边眉毛 21 ~ 26
            //鼻梁     27 ~ 30
            //鼻孔        31 ~ 35
            //左眼        36 ~ 41
            //右眼        42 ~ 47
            //嘴唇外圈  48 ~ 59
            //嘴唇内圈  59 ~ 67
            switch(i)
            {
                case 16:
                case 21:
                case 26:
                case 30:
                case 35:
                case 41:
                case 47:
                case 59:
                    i++;
                    break;
                default:
                    break;
            }
 
            p1.x = fs[j].part(i).x();
            p1.y = fs[j].part(i).y();
            p2.x = fs[j].part(i+1).x();
            p2.y = fs[j].part(i+1).y();
            //cv::line(img, p1, p2, cv::Scalar(0,0,255), 2, 4, 0);
            cv::circle(img, p1, 3, cv::Scalar(0, 0, 255), -1);
	}
    }
}
 
 
int main(int argc, char *argv[])
{
    cv::VideoCapture cap("../test.mp4");

    if( !cap.isOpened() ){
    	std::cerr<<"Can not open the video !"<<std::endl;
	return 0;
    }

    //加载dlib的人脸识别器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
 
    //加载人脸形状探测器
    dlib::shape_predictor sp;
    dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> sp;

    cv::Mat frame;
    while( 1 ) {
    	cap >> frame;

	cv::cvtColor(frame, frame, CV_BGR2GRAY);

	//cv::resize( frame, frame, cv::Size(640, 360) );

	//Mat转化为dlib的matrix
    	dlib::array2d<dlib::bgr_pixel> dimg;
    	dlib::assign_image(dimg, dlib::cv_image<uchar>(frame));
 
	    //获取一系列人脸所在区域
    	//auto t1 = std::chrono::steady_clock::now();
    	std::vector<dlib::rectangle> dets = detector(dimg);
    	std::cout << "Number of faces detected: " << dets.size() << std::endl;

   // 	if (dets.size() == 0) continue;


    	//获取人脸特征点分布
	auto t1 = std::chrono::steady_clock::now();

    	std::vector<dlib::full_object_detection> shapes;
    	int i = 0;
    	for(i = 0; i < dets.size(); i++) {
        	dlib::full_object_detection shape = sp(dimg, dets[i]); //获取指定一个区域的人脸形状
        	shapes.push_back(shape);
    	}

	//指出每个检测到的人脸的位置
    	for(i=0; i<dets.size(); i++){
        	//画出人脸所在区域
	        cv::Rect r;
        	r.x = dets[i].left();
        	r.y = dets[i].top();
        	r.width = dets[i].width();
        	r.height = dets[i].height();
        	cv::rectangle(frame, r, cv::Scalar(0, 0, 255), 1, 1, 0);
    	}

    	line_one_face_detections(frame, shapes);
    
    	auto t2 = std::chrono::steady_clock::now();
	double dr_ms = std::chrono::duration<double,std::milli>(t2-t1).count();
        std::cout<<"duration : "<<dr_ms<<std::endl;
    	
    	cv::imshow("frame", frame);
    	
	if( cv::waitKey(20) == 'q' ) break;
    }

 

    return 0;
}
