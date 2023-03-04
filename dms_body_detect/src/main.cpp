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

#include <thread>
#include <mutex>
#include <condition_variable>

#include "body_detect.h"
#include <chrono>

#define IMAGE_SIZE 480 * 272


typedef struct BBox_
{
	int head_box[4] = {0};
	int face_box[4] = {0};
}BBox;

std::mutex mtx;
std::condition_variable con;

cv::Mat src;
BBox boxes;
bool box_flag = false;


const int initUdpClient()
{
        int sockfd = socket( AF_INET, SOCK_DGRAM, 0 );
        if ( sockfd <= 0 ) {
                std::cerr<<"socket UDP Client init failed !"<<std::endl;
                return false;
        }

        std::cout<<"Init the Udp Client !"<<std::endl;

        return sockfd;
}

void threadCamera()
{
	// 1. body detection
        body_detect::BodyDetection detect;

	// 2. start the camera
        cv::VideoCapture cap(8);
        if ( !cap.isOpened() ) {
                std::cout<<"Camera Open Failed !"<<std::endl;
                return;
        }
        std::cout<<"Open the Camera !"<<std::endl;

        cv::Mat frame;
        int count = 0;
        while ( 1 ) {
                cap >> frame;

		if ( frame.empty() ) break;
                cv::resize( frame, frame, cv::Size( 960, 544 ) );

                cv::imshow("video", frame);
	
		auto begin = std::chrono::steady_clock::now();
                detect.imageProcess( frame );
                auto end = std::chrono::steady_clock::now();

                auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
                std::cout<<"detect duration : "<<dur<<" ms"<<std::endl;

		// box results
		auto rois_ret = detect.getRoisRet();
		{
			std::unique_lock<std::mutex> lck( mtx );
			if ( rois_ret[5].size() == 1 && rois_ret[3].size() == 1 ) {
				boxes.head_box[0] = rois_ret[3][0].x;
				boxes.head_box[1] = rois_ret[3][0].y;
				boxes.head_box[2] = rois_ret[3][0].width;
				boxes.head_box[3] = rois_ret[3][0].height;

				boxes.face_box[0] = rois_ret[5][0].x;
        	                boxes.face_box[1] = rois_ret[5][0].y;
                	        boxes.face_box[2] = rois_ret[5][0].width;
	                        boxes.face_box[3] = rois_ret[5][0].height;
				
				box_flag = true;
			}

               	 	cv::resize( frame, src, cv::Size( 480, 272 ) );
                	con.notify_one();
                }
 		
		if ( cv::waitKey( 5 ) == 'q' ) break;
 	}

}

void threadTransport()
{
        int sockfd = initUdpClient();

        struct sockaddr_in dst_addr;
        dst_addr.sin_family = AF_INET;
        dst_addr.sin_addr.s_addr = inet_addr( "127.0.0.1" );
        dst_addr.sin_port = htons( 2333 );

        while ( 1 ) {
		auto begin = std::chrono::steady_clock::now();
                // 1. encode and send
                std::vector<unsigned char> encode_data;
                {
                std::unique_lock<std::mutex> lck( mtx );
                con.wait( lck );

                cv::cvtColor( src, src, cv::COLOR_BGR2GRAY );

                cv::imencode(".jpg", src, encode_data);
                }

                int ret = sendto( sockfd, encode_data.data(), encode_data.size(), 0, (struct sockaddr*)&dst_addr, sizeof(dst_addr) );
                if ( ret < 0 ) {
                        std::cerr<<"send failed !"<<std::endl;
                }
                else {
                        std::cout<<"send "<<ret<<" bytes data"<<std::endl;
                }

		if ( box_flag == true ) {
			ret = sendto( sockfd, &boxes, sizeof(boxes), 0, (struct sockaddr*)&dst_addr, sizeof(dst_addr) );
        	        if ( ret < 0 ) {
                	        std::cerr<<"send failed !"<<std::endl;
                	}
	                else {
        	                std::cout<<"send "<<ret<<" bytes data"<<std::endl;
                	}
		}


                auto end = std::chrono::steady_clock::now();

                auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
                std::cout<<"send duration : "<<dur<<" ms"<<std::endl;

        }
}

int main()
{
	std::cout<<"---------------- VIDEO BODY DETECT ---------------"<<std::endl;
	std::thread t1( threadCamera );
        std::thread t2( threadTransport );

        t1.join();
        t2.join();

	return 0;
}
