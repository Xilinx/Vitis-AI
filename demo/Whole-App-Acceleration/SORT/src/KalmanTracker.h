///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H 2

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

#define StateType Rect_<float>


// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
	KalmanTracker(StateType initRect) : m_time_since_update(0), m_hits(0), m_hit_streak(0), m_age(0),
		state(cv::Mat(cv::Size(1, 7), CV_32FC1, Scalar(0))), // x
		processNoiseCov(cv::Mat(cv::Size(7, 7), CV_32FC1, Scalar(0))), // Q
		measurementNoiseCov(cv::Mat(cv::Size(4, 4), CV_32FC1, Scalar(0))), //R
		errorCov(cv::Mat(cv::Size(7, 7), CV_32FC1, Scalar(0))), // P
		errorCovD(cv::Mat(cv::Size(1, 7), CV_32FC1, Scalar(0))), // P
		measurementMatrix(cv::Mat(cv::Size(4, 7), CV_32FC1, Scalar(0))) // H
	{
		init_kf(initRect);
		m_id = kf_count++;
	}

	StateType getRect() const;
	void printState() const;
	void printRect() const;
	const cv::Mat& getState() const { 
	//	if(state.isContinuous())
			return state;
	//	else
	//		return state.clone();
    } 
	const cv::Mat& getErrorCov() const { return errorCov; }
	const cv::Mat& getErrorCovD() const { return errorCovD; }

	// void setState(const cv::Mat state) { this->state = state; }
	// void setErrorCov(const cv::Mat errorCov) { this->errorCov = errorCov; }

	void setState(float* data) {
		memcpy(state.data, data, 7*sizeof(float));
	}
	void setErrorCov(float* data) {
		memcpy(errorCov.data, data, 7*7*sizeof(float));
	}
	void setErrorCovD(float* data) {
		memcpy(errorCovD.data, data, 7*sizeof(float));
	}

	static int kf_count;

	int m_time_since_update;
	int m_hits;
	int m_hit_streak;
	int m_age;
	int m_id;

private:
	cv::Mat transitionMatrix;  // (7x7)
	cv::Mat measurementMatrix;  // (4, 7)
	cv::Mat processNoiseCov;  // (7, 7)
	cv::Mat measurementNoiseCov;  // (4, 4)

	cv::Mat state;  // [cx, cy, a, r, v_cx, v_cy, v_a] (7, 1)
	cv::Mat errorCov;  // (7, 7)
	cv::Mat errorCovD;  // (1, 7)

	void init_kf(StateType stateMat);

};

#endif
