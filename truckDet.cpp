
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <queue>
#include <mutex>
#include <thread>
#include <chrono>

#include <ctime>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char * classNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };


const string CAMERA_PATH_0 = "rtsp://admin:Freedom!00##@192.168.101.21:554/ISAPI/Streaming/Channels/101" ;
const string CAMERA_PATH_1 = "rtsp://admin:Freedom!00##@192.168.101.22:554/ISAPI/Streaming/Channels/101";
const string FTP_PATH = "/home/cam/files/truckArchive/" ;
const string MODEL_CONFIG = "/home/roniinx/video_cv/truckDetection/models/SSD/ssd.prototxt" ;
const string MODEL_BINARY = "/home/roniinx/video_cv/truckDetection/models/SSD/ssd.caffemodel" ;
const double CONFIDENCE_THRESHOLD = 0.75;
const bool USE_ROI = false;
const bool DETECT_PLATE = false;
const string HAAR_CASCADE_PATH = "/home/roniinx/video_cv/truckDetection/models/haarcascade_russian_plate_number.xml";
const int DETECTION_PERIOD = 30;


//  ROI - fraction of input Mat 
struct ROI_rectangle {
	double dx;
	double dy;
	double lx;
	double ly;
};

const ROI_rectangle ROI_RECT{ 0.1, 0.2, 0.8, 0.6 };

queue<Mat> camera0;
mutex mutex_camera0;

queue<Mat> camera1;
mutex mutex_camera1;


Mat GetMatCamera(int cam_id) {

	if (cam_id == 0) {

		mutex_camera0.lock();

		if ( camera0.size() > 0 ) {

			Mat ret_mat = camera0.front().clone();

			while ( !camera0.empty() ) {
				camera0.pop();
			}

			mutex_camera0.unlock();
			return ret_mat;
		}
		else {

			mutex_camera0.unlock();
			return Mat();
		}
	}
	else if (cam_id == 1) {

		mutex_camera1.lock();

		if (camera1.size() > 0) {

			Mat ret_mat = camera1.front().clone();

			while (!camera1.empty()) {
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


		if (cam_id == 0) {

			mutex_camera0.lock();
			camera0.push(frame);
			mutex_camera0.unlock();
		}
		else if (cam_id == 1) {

			mutex_camera1.lock();
			camera1.push(frame);
			mutex_camera1.unlock();
		}

	}
}


void DetectCar(int cam_id) {

	String modelConfiguration = MODEL_CONFIG;
	String modelBinary = MODEL_BINARY;

	dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);

	cv::CascadeClassifier plateCascade;

	if (!plateCascade.load(HAAR_CASCADE_PATH)) {
		std::cout << "Error when loading the face cascade classfier!" << std::endl;
		exit(EXIT_FAILURE);
	}


	long long lastWritten = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count() - DETECTION_PERIOD;

	while (true) {

		Mat frame = GetMatCamera(cam_id);

		if (!frame.data) {
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			continue;
		}

		Mat roi_frame = frame;

		if (USE_ROI) {

			roi_frame = frame(Rect(frame.cols * ROI_RECT.dx, frame.rows * ROI_RECT.dy, frame.cols * ROI_RECT.lx, frame.rows * ROI_RECT.ly));

			rectangle(frame, Rect(frame.cols * ROI_RECT.dx, frame.rows * ROI_RECT.dy, frame.cols * ROI_RECT.lx, frame.rows * ROI_RECT.ly),
				Scalar(0, 255, 255), 1);

			putText(frame, "ROI", Point{ (int)(frame.cols * ROI_RECT.dx), (int)(frame.rows * ROI_RECT.dy) }, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
		}


		Mat inputBlob = blobFromImage(roi_frame, 1 / 255.F,
			Size(300, 300),
			Scalar(127, 127, 127),
			true, false);

		net.setInput(inputBlob);

		//double t = (double)getTickCount();

		Mat detection = net.forward();

		//t = ((double)getTickCount() - t) / getTickFrequency();
		//cout << "net forward time elapsed = " << t << endl;


		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		float confidenceThreshold = CONFIDENCE_THRESHOLD;

		bool is_car_detected = false;
		bool is_plate_detected = false;

		for (int i = 0; i < detectionMat.rows; ++i) {

			size_t objectClass = (size_t)detectionMat.at<float>(i, 1);

			float confidence = detectionMat.at<float>(i, 2);

			if (classNames[objectClass] == "bus" || classNames[objectClass] == "car") {

				if (confidence > confidenceThreshold) {

					is_car_detected = true;

					int left = static_cast<int>(detectionMat.at<float>(i, 3) * roi_frame.cols);
					int top = static_cast<int>(detectionMat.at<float>(i, 4) * roi_frame.rows);
					int right = static_cast<int>(detectionMat.at<float>(i, 5) * roi_frame.cols);
					int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * roi_frame.rows);

					rectangle(roi_frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);

					

					if (DETECT_PLATE) {


						Mat vehicle_frame = roi_frame(
							Rect(
								max(0, left),
								max(0, top),
								min(right - left, roi_frame.cols - max(0, left)),
								min(bottom - top, roi_frame.rows - max(0, top))
							)
						);


						vector<Rect> detections;

						//double t1 = (double)getTickCount();

						plateCascade.detectMultiScale(vehicle_frame,
							detections,
							1.1,
							3);

						//t1 = ((double)getTickCount() - t1) / getTickFrequency();
						//cout << "detectMultiScale time elapsed = " << t1 << endl;

						if (detections.size() > 0) {
							is_plate_detected = true;
						}

						for (int i = 0; i < detections.size(); ++i) {
							rectangle(vehicle_frame, detections[i], cv::Scalar(0, 0, 255), 2);
						}

					}

					
					


				}

			}

		}


		long long now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();


		if (is_car_detected && (!DETECT_PLATE || is_plate_detected) && now - lastWritten > DETECTION_PERIOD) {

			lastWritten = now;

			std::time_t t = std::time(nullptr);
			char mbstr[100];
			std::strftime(mbstr, sizeof(mbstr), "%Y%m%d%H%M%S000", std::localtime(&t));
			string filenameToRecord = FTP_PATH + "A100BC50" + "_" + mbstr + "_" + to_string(cam_id) + ".jpg";

			cv::imwrite(filenameToRecord, frame);

			std::this_thread::sleep_for(std::chrono::seconds(DETECTION_PERIOD));
		}

	}

}


int main() {

	thread thread_camera_0 = thread(ConnectCamera, CAMERA_PATH_0, 0);
	thread_camera_0.detach();

	thread thread_dnn_0 = thread(DetectCar, 0);
	thread_dnn_0.detach();

	/*
	thread thread_camera_1 = thread(ConnectCamera, CAMERA_PATH_1, 1);
	thread_camera_1.detach();

	thread thread_dnn_1 = thread(DetectCar, 1);
	thread_dnn_1.detach();
	
	*/



	while (true) {

		this_thread::sleep_for(chrono::seconds(60));
	}

	return 0;
}