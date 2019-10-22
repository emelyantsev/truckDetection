
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <queue>
#include <mutex>
#include <thread>
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char * classNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };


const string CAMERA_PATH_1 = "rtsp://admin:Freedom!00##@192.168.106.20:554/h264/ch1/sub/av_stream" ;
const string FTP_PATH = "/home/roniinx/ftp/truckArchive/" ;
const string MODEL_CONFIG = "/home/roniinx/video_cv/truckDetection/release/SSD/ssd.prototxt" ;
const string MODEL_BINARY = "/home/roniinx/video_cv/truckDetection/release/SSD/ssd.caffemodel" ;
const double CONFIDENCE_THRESHOLD = 0.75;


queue<Mat> camera1;
mutex mutex_camera1;


Mat GetMatCamera(int cam_id) {

	if (cam_id == 1) {

		mutex_camera1.lock();

		if ( camera1.size() > 0 ) {

			Mat ret_mat = camera1.front().clone();

			while ( !camera1.empty() ) {
				camera1.pop();
			}

			mutex_camera1.unlock();
			return ret_mat;
		}
		else {

			mutex_camera1.unlock();
			return Mat();
		}
	}

	return Mat();
}

void ConnectCamera(string path, int cam_id)
{
	int badFrameCounter = 0;

	VideoCapture cap;
	cap = VideoCapture(path);

	while (!cap.isOpened()) {
		std::this_thread::sleep_for(std::chrono::seconds(600));
		cap = VideoCapture(path);
	}

	while (true) {

		Mat frame;
		cap >> frame;

		if (!frame.data) {

			badFrameCounter++;

			if (badFrameCounter > 5) {

				badFrameCounter = 0;
				cap.release();

				cap = VideoCapture(path);

				while (!cap.isOpened()) {
					std::this_thread::sleep_for(std::chrono::seconds(600));
					cap = VideoCapture(path);
				}
	
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			continue;
		}

		badFrameCounter = 0;

		if (cam_id == 1) {

			mutex_camera1.lock();
			camera1.push(frame);
			mutex_camera1.unlock();
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}
}


void DetectCar(int cam_id) {

	String modelConfiguration = MODEL_CONFIG;
	String modelBinary = MODEL_BINARY;

	dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);

	int lastWritten = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	while (true) {

		Mat frame = GetMatCamera(cam_id);

		if (!frame.data) {
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			continue;
		}


		Mat inputBlob = blobFromImage(frame, 1 / 255.F,
										Size(300, 300),
										Scalar(127, 127, 127),
										true, false);

		net.setInput(inputBlob);

		//double t = (double) getTickCount() ;

		Mat detection = net.forward();

		//t = ((double) getTickCount() - t) / getTickFrequency();
		//cout << "time elapsed = " << t << endl;


		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		float confidenceThreshold = CONFIDENCE_THRESHOLD;

		for (int i = 0; i < detectionMat.rows; i++) {

			size_t objectClass = (size_t) detectionMat.at<float>(i, 1);
			
			float confidence = detectionMat.at<float>(i, 2);
			
			if (classNames[objectClass] == "bus" || classNames[objectClass] == "car") {

				if (confidence > confidenceThreshold) {

					int left = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
					int top = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
					int right = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
					int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

					rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 5);


					int now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();


					if (now - lastWritten > 30) {

						lastWritten = now;
						string filenameToRecord = FTP_PATH + to_string(now) + string(classNames[objectClass]) + string(".jpg");
						cv::imwrite(filenameToRecord, frame);
					}
				}

			}
		}
	}

}



int main() {

	thread thread_camera_1 = thread(ConnectCamera, CAMERA_PATH_1, 1);
	thread_camera_1.detach();

	thread thread_dnn_1 = thread(DetectCar, 1);
	thread_dnn_1.detach();


	while (true) {

		this_thread::sleep_for(chrono::seconds(60));
	}

	return 0;
}