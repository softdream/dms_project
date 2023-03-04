#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <sys/shm.h>

#define IMAGE_SIZE 480 * 272

typedef struct Box_
{
        uint8_t flag = 0;

        char img_data[IMAGE_SIZE] = {0};
}Box;


int main(int argc, char *argv[])
{
	std::cout<<"------------------ FACE Features -----------------"<<std::endl;
	
	// 1. shape predictor 
	//dlib::shape_predictor sp;
        //dlib::deserialize("/home/sunrise/workstation/dms_projects/dms_project/dnn_body_detect/model/shape_predictor_68_face_landmarks.dat") >> sp;

        std::cout<<"Shape Predictor Loaded !"<<std::endl;		

	// 1. shared memory
        int shmid = shmget( (key_t)1275, sizeof(Box), 0666|IPC_CREAT );
        void *shm = shmat( shmid, (void *)0, 0 );
        Box *p_box = (Box *)shm;

        std::cout<<"start "<<std::endl;
        p_box->flag = 0;
        while ( 1 ) {
                if ( p_box->flag == 1 ) {
                        //cv::Mat frame = cv::Mat( 272, 480, CV_8UC3, cv::Scalar(255, 255, 255) );
                        cv::Mat frame = cv::Mat( 272, 480, CV_8UC1 );

                        memcpy( (char *)frame.data, p_box->img_data, IMAGE_SIZE );

                        cv::imshow("frame", frame);
                        cv::waitKey(10);

                        p_box->flag = 0;
                }
        }

        shmdt( shm );
        shmctl( shmid, IPC_RMID, 0 );


    	return 0;
}
