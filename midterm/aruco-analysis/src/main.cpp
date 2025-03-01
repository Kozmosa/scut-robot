#include <opencv2\opencv.hpp>
#include <iostream>

int main() {
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Cannot open camera." << std::endl;
		return -1;
	}

	while (true) {
		// infinite loop reading frame from the camera
		cv::Mat frame;
		cap >> frame;
		if (frame.empty()) {
			std::cerr << "Cannot read frame" << std::endl;
			break;
		}

		cv::imshow("Aruco Analysis Camera 0", frame);
	}

	return 0;
}