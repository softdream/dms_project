#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <malloc.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <stdarg.h>
#include <fcntl.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <dlib/opencv.h>
#include "opencv2/opencv.hpp"

#include <vector>

#define IMAGE_SIZE 480 * 272

//unsigned char buff[IMAGE_SIZE] = { 0 };
std::vector<unsigned char> buff( IMAGE_SIZE, 0 );

typedef struct BBox_
{
        int head_box[4] = {0};
        int face_box[4] = {0};
}BBox;

BBox boxes;
cv::Mat image = cv::Mat( 272, 480, CV_8UC1 );

static const dlib::rectangle cvRect2Dlib( const cv::Rect& r )
{
    return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

static void drawFaceKeyPoints( cv::Mat& img, const dlib::full_object_detection& fs )
{
	for ( int i = 0; i < 68; i ++ ) {
		// 下巴到脸颊 0 ~ 16
            	//左边眉毛 17 ~ 21
            	//右边眉毛 21 ~ 26
            	//鼻梁     27 ~ 30
            	//鼻孔        31 ~ 35
            	//左眼        36 ~ 41
            	//右眼        42 ~ 47
            	//嘴唇外圈  48 ~ 59
            	//嘴唇内圈  59 ~ 67

		cv::Point pt( fs.part(i).x(), fs.part(i).y() );

		cv::circle(img, pt, 2, cv::Scalar(0, 0, 255), -1);
	}
}

const int initUdpServer()
{
	int sockfd = socket( AF_INET, SOCK_DGRAM, 0 );
	if ( sockfd <= 0 ) {
		std::cerr<<"socket UDP Server init failed !"<<std::endl;
		return false;
	}

	struct sockaddr_in srv_addr;
	srv_addr.sin_family = AF_INET;
	srv_addr.sin_addr.s_addr = htonl( INADDR_ANY );
	srv_addr.sin_port = htons( 2333 );

	if ( bind( sockfd, (struct sockaddr *)&srv_addr, sizeof( srv_addr ) ) < 0 ) {
		std::cerr<<"Failed to bind the socket server addr !"<<std::endl;
		return -1;
	}

	std::cout<<"Init the Udp Server !"<<std::endl;

	return sockfd;
}

int main()
{
	std::cout<<"--------------- OPENCV VIDEO TEST ---------------"<<std::endl;
	int sockfd = initUdpServer();
	struct sockaddr_in clt_addr;
	socklen_t clt_len;

	//加载人脸形状探测器
    	dlib::shape_predictor sp;
   	dlib::deserialize("./shape_predictor_68_face_landmarks.dat") >> sp;
	std::cout<<"loaded the shape predictor file !"<<std::endl;

	while ( 1 ) {
		int ret = recvfrom( sockfd, buff.data(), buff.size(), 0, (struct sockaddr *)&clt_addr, &clt_len );
		if ( ret <= 0 ) {
			std::cerr<<"recv failed !"<<std::endl;
		}
		else if ( ret == 32 ) {
			memcpy( &boxes, buff.data(), sizeof( boxes ) );
		}
		else {
			image = cv::imdecode(buff, cv::IMREAD_COLOR);//图像解码

			cv::Rect head_rect( boxes.head_box[0] / 2, boxes.head_box[1] / 2, boxes.head_box[2] / 2, boxes.head_box[3] / 2 );
                        cv::Rect face_rect( boxes.face_box[0] / 2, boxes.face_box[1] / 2, boxes.face_box[2] / 2, boxes.face_box[3] / 2 );

                        std::cout<<"face rect : "<<boxes.face_box[0] / 2 <<", "<<boxes.face_box[1] / 2<<", "<<boxes.face_box[2] / 2<<", "<<boxes.face_box[3] / 2<<std::endl;
		
			auto begin = std::chrono::steady_clock::now();
			// detect
			if ( boxes.face_box[0] && boxes.face_box[1] && boxes.face_box[2] && boxes.face_box[3] ) {
	
				cv::Mat img;
				cv::cvtColor( image, img, cv::COLOR_BGR2GRAY );

				dlib::array2d<dlib::bgr_pixel> dimg;
	        		dlib::assign_image(dimg, dlib::cv_image<uchar>( img ));
				
				dlib::full_object_detection shape = sp( dimg, cvRect2Dlib( face_rect ) );
				drawFaceKeyPoints( image, shape );
			
			}

			auto end = std::chrono::steady_clock::now();

                	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
                	std::cout<<"face feature duration : "<<dur<<" ms"<<std::endl;

			cv::rectangle( image, head_rect, cv::Scalar(255, 0, 0), 2 );
                        cv::rectangle( image, face_rect, cv::Scalar(255, 0, 0), 2 );

			cv::imshow("image", image);
			cv::waitKey(5);
			memset( &boxes, 0, sizeof( boxes ) );
		}
	}

	return 0;
}
