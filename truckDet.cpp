

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>


using namespace std;
using namespace cv;
using namespace cv::dnn;


static const char* classNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

//read-mage cam_test modion detect
struct ObjTypePeopleDetect
{
    //person-car
    string type_object;
    cv::Rect rect_object;
};

static vector<Mat> camera1;
//static vector<Mat> camera2;
static mutex mutex_camera1;
//static mutex mutex_camera2;



Mat GetMatCamera(int cam_id)
{
    if (cam_id == 1)
    {
        if (camera1.size() > 0)
        {
            //cout << "size1:" << camera1.size() << endl;
            mutex_camera1.lock();
            Mat ret_mat = camera1.front().clone();
            if (camera1.size() > 10)
            {
                camera1.clear();
            }
            mutex_camera1.unlock();
            return ret_mat;
        }
        else return Mat();
    }
    else return Mat();
 /*   else if (cam_id == 2)
    {
        if (camera2.size() > 0)
        {
            cout << "size2:" << camera2.size() << endl;
            mutex_camera2.lock();
            Mat ret_mat = camera2.front().clone();
            if (camera2.size() > 10)
            {
                camera2.clear();
            }
            mutex_camera2.unlock();

            return ret_mat;
        }
    }*/
}

void ConnectCamera(string path, int cam_id)
{
    int badFrameCounter = 0;
    VideoCapture cap;
    cap = VideoCapture(path);

    if (!cap.isOpened())
    {
        return;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;
        //cout << "gotFrame " << frame.rows << " " << frame.cols<< "\n";
        if (!frame.data)
        {
            badFrameCounter++;
            if(badFrameCounter > 5){
                badFrameCounter = 0;
                cap.release();
                cap = VideoCapture(path);
                if (!cap.isOpened())
                {
                    return;
                }
            }
             std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        badFrameCounter = 0;
        if (cam_id == 1)
        {

                mutex_camera1.lock();
                camera1.push_back(frame);
                mutex_camera1.unlock();


        }
     /*   else if (cam_id == 2)
        {

            mutex_camera2.lock();
            camera2.push_back(frame);
            mutex_camera2.unlock();

        }*/

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void DetectCar(int cam_id)
{
    String modelConfiguration = "/home/roniinx/video_cv/truckDetection/release/SSD/ssd.prototxt";
    String modelBinary = "/home/roniinx/video_cv/truckDetection/release/SSD/ssd.caffemodel";
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);

    string name_cam = "Plate_" + to_string(cam_id);
    //namedWindow(name_cam);
    long lastWritten = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    while (1)
    {
        Mat frame = GetMatCamera(cam_id);
        if (!frame.data)
        {
            // << "frame empty \n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;

        }
           // cout << "frame NOTempty " << frame.rows << " " << frame.cols<< "\n";

        frame = frame(Rect(0, frame.rows/5,frame.cols - 1, frame.rows *4/5 -1));
        Mat frame1 = frame.clone();

        //cout << "frame resized \n";
        Mat inputBlob = blobFromImage(frame, 1 / 255.F,
            Size(180, 180),
            Scalar(127, 127, 127),
            true, false);

    //cout << "BLOB DONE \n";
        net.setInput(inputBlob);

        Mat detection = net.forward();


        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());


        float confidenceThreshold = 0.5;//0.2
        for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
            if (classNames[objectClass] == "bus" || classNames[objectClass] == "car")
            {
                //cout << "conf " << classNames[objectClass] << " " << confidence << "\n";
                if (confidence > confidenceThreshold)
                {
                    //size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

                    int left = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                    int top = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                    int right = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                    int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 5);
                    long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    if(lastWritten - now > 3000){
                        lastWritten = now;
                        string filenameToRecord1 = "/home/roniinx/ftp/bagArchive/" + to_string(now) + "�ar.jpg";
                        cv::imwrite(filenameToRecord1, frame);
                    }
                }
            }
        }


        /*cv::imshow(name_cam, frame);
        waitKey(500);*/
    }
}

int main()
{


    //cout << "gggggg";
    thread thread_camera_1 = thread(ConnectCamera, "rtsp://admin:Freedom!00##@192.168.106.20:554/h264/ch1/sub/av_stream", 1);
    thread_camera_1.detach();

    //thread thread_camera_2 = thread(ConnectCamera, "rtsp://admin:Freedom!00##@192.168.104.20:554/h264/ch1/sub/av_stream", 2);
    //thread_camera_2.detach();

    thread thread_dnn_1 = thread(DetectCar, 1);
    thread_dnn_1.detach();

    //thread thread_dnn_2 = thread(DetectCar, 2);
    //thread_dnn_2.detach();




    for (;;)
    {

        std::this_thread::sleep_for(std::chrono::milliseconds(15000));

    }


}



