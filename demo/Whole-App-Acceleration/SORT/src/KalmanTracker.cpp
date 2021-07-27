///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "KalmanTracker.h"
#include <iostream>


int KalmanTracker::kf_count = 0;


// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	const int stateNum = 7;
	const int measureNum = 4;

	// state = cv::Mat(cv::Size(stateNum, 1), CV_32FC1, Scalar(0));  // x
	// processNoiseCov = cv::Mat(cv::Size(stateNum, stateNum), CV_32FC1, Scalar(0)); // Q
	// measurementNoiseCov = cv::Mat(cv::Size(measureNum, measureNum), CV_32FC1, Scalar(0)); // R
	// errorCov = cv::Mat(cv::Size(stateNum, stateNum), CV_32FC1, Scalar(0)); // P
	// measurementMatrix = cv::Mat(cv::Size(measureNum, stateNum), CV_32FC1, Scalar(0)); // H

	// A or F
	transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);
	setIdentity(measurementMatrix);
	setIdentity(processNoiseCov, Scalar::all(1e-2));
	setIdentity(measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(errorCov, Scalar::all(1));
	errorCovD = cv::Mat::ones(1, 7, CV_32FC1);

	state.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	state.at<float>(0, 1) = stateMat.y + stateMat.height / 2;
	state.at<float>(0, 2) = stateMat.area();
	state.at<float>(0, 3) = stateMat.width / stateMat.height;
}

void KalmanTracker::printState() const {
	
	std::cout << "cx: " << state.at<float>(0, 0)
			  << " | cy: " << state.at<float>(0,1)
			  << " | s: " << state.at<float>(0,2)
			  << " | r: " << state.at<float>(0,3)
			  << std::endl;
	//std::cout << "state " << state << std::endl;
				  
}

void KalmanTracker::printRect() const {
	auto rect = getRect();
	std::cout << "x: " << rect.x
			  << " | y: " << rect.y
			  << " | w: " << rect.width
			  << " | h: " << rect.height
			  << std::endl;
}


// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::getRect() const
{
	float cx = state.at<float>(0, 0);
	float cy = state.at<float>(0, 1);
	float s = state.at<float>(0, 2);
	float r = state.at<float>(0, 3);

	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;
	return StateType(x, y, w, h);
}
