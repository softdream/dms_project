#include "body_detect.h"
#include <chrono>

int main()
{
	std::cout<<"---------------- BODY DETECT ---------------"<<std::endl;

	body_detect::BodyDetection detect;

	/*cv::Mat image = cv::imread("/home/sunrise/workstation/dnn_projects/sunrise_X3_BPU/dnn_body_detect/test_data/1.jpg");

	cv::resize( image, image, cv::Size( 960, 544 ) );

	cv::imshow( "test", image );
	cv::waitKey(0);

	auto begin = std::chrono::steady_clock::now();
	detect.imageProcess( image );
	auto end = std::chrono::steady_clock::now();

	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout<<"duration : "<<dur<<" ms"<<std::endl;

	detect.displayResults( image );
	cv::imshow( "ret", image );
	cv::waitKey(0);

	cv::imwrite("1_ret.jpg", image);
	*/

	cv::VideoCapture cap(8);
        if ( !cap.isOpened() ) {
                std::cout<<"Camera Open Failed !"<<std::endl;
                return 0;
        }
        std::cout<<"Open the Camera !"<<std::endl;

	cv::Mat frame;
        while ( 1 ) {
                cap >> frame;
                if ( frame.empty() ) break;

		cv::resize( frame, frame, cv::Size( 960, 544 ) );

		auto begin = std::chrono::steady_clock::now();
        	detect.imageProcess( frame );
        	auto end = std::chrono::steady_clock::now();

        	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        	std::cout<<"duration : "<<dur<<" ms"<<std::endl;

        	detect.displayResults( frame );
		cv::imshow("body detect", frame);
                if ( cv::waitKey( 5 ) == 'q' ) break;
        }

	return 0;
}
