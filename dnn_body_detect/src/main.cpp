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

#include "body_detect.h"


using namespace std;

static const dlib::rectangle openCVRect2Dlib(const cv::Rect& r)
{
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

void drawFaceKeyPoints(cv::Mat& img, const std::vector<dlib::full_object_detection>& fs)
{
    int i, j;
    for ( j = 0; j < fs.size(); j ++ ) {
        cv::Point p1, p2;
        for( i = 0; i < 67; i ++ ) {
            // 下巴到脸颊 0 ~ 16
            //左边眉毛 17 ~ 21
            //右边眉毛 21 ~ 26
            //鼻梁     27 ~ 30
            //鼻孔        31 ~ 35
            //左眼        36 ~ 41
            //右眼        42 ~ 47
            //嘴唇外圈  48 ~ 59
            //嘴唇内圈  59 ~ 67
	    switch(i) {
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
	std::cout<<"---------------- BODY DETECT ---------------"<<std::endl;
	body_detect::BodyDetection detect;

	cv::VideoCapture cap(8);

    	if( !cap.isOpened() ){
        	std::cerr<<"Can not open the video !"<<std::endl;
        	return 0;
    	}
	std::cout<<"Open the Camera !"<<std::endl;

    	//加载人脸形状探测器
    	dlib::shape_predictor sp;
    	dlib::deserialize("/home/sunrise/workstation/dms_projects/dms_project/dnn_body_detect/model/shape_predictor_68_face_landmarks.dat") >> sp;

	std::cout<<"Shape Predictor Loaded !"<<std::endl;

    	cv::Mat frame, src;

	while( 1 ) {
        	cap >> src;
		if ( src.empty() ) break;

        	cv::resize( src, src, cv::Size(960, 544) );

		detect.imageProcess( src );

		auto rois_ret = detect.getRoisRet();

		cv::cvtColor(src, frame, cv::COLOR_BGR2GRAY);
        	dlib::array2d<dlib::bgr_pixel> dimg;
        	dlib::assign_image(dimg, dlib::cv_image<uchar>(frame));


		std::vector<dlib::rectangle> dets;

		bool has_face = false;
		for ( const auto& obs : rois_ret ) {
			if ( obs.first == 5 ) {  // face
				for ( const auto& rect : obs.second ) {
					dets.push_back( openCVRect2Dlib( rect ) );
				}

				has_face = true;
			}
		}

		if ( has_face ) {
			auto begin = std::chrono::steady_clock::now();
			std::vector<dlib::full_object_detection> shapes;
        	
			for( int i = 0; i < dets.size(); i++) {
                		dlib::full_object_detection shape = sp(dimg, dets[i]);
                		shapes.push_back(shape);
        		}

			drawFaceKeyPoints( src, shapes );

			auto end = std::chrono::steady_clock::now();

		        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
			std::cout<<"duration : "<<dur<<" ms"<<std::endl;
		}

		detect.displayResults( src );
                cv::imshow("body detect", src);
                if ( cv::waitKey( 2 ) == 'q' ) break;
	}


    	return 0;
}

