///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
// #include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <unistd.h>
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "profiling.hpp"

#include "kf_xrt/kf_wrapper.h"

using namespace std;
using namespace cv;
using namespace vitis::ai;


typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
} TrackingBox;


// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}


// global variables for counting
int total_frames = 0;
double total_time = 0.0;
// global hardware KF accelerator enable
bool hw_kf_en;

void TestSORT(string inputDir, bool display);


int main(int argc, char **argv)
{
	if (argc != 3) {
        std::cout << "Usage: .exe <inputDir> <waa en>" << std::endl;
    }
    const std::string inputDir = argv[1];

	if(atoi(argv[2])==0)
           hw_kf_en = 0;
	else
	   hw_kf_en = 1;

	TestSORT(inputDir, false);

	// Note: time counted here is of tracking procedure, while the running speed bottleneck is opening and parsing detectionFile.
	cout << "Total Tracking took: " << total_time << " for " << total_frames << " frames or " << ((double)total_frames / (double)total_time) << " FPS" << endl;

	return 0;
}

//global buffer for states and covariances
float* states_in_global = (float*)malloc(500 * 7* sizeof(float));
float* covariances_in_global = (float*)malloc(500 * 7 * 7* sizeof(float));
float* covariancesD_in_global = (float*)malloc(500 * 7* sizeof(float));
float* states_out_global = (float*)malloc(500 * 7* sizeof(float));
float* covariances_out_global = (float*)malloc(500 * 7 * 7* sizeof(float));
float* covariancesD_out_global = (float*)malloc(500 * 7* sizeof(float));
float* measurements_global = (float*)malloc(500 * 4* sizeof(float));

void sw_predict(cv::KalmanFilter kf_sw, size_t num_kf) 
{
	for(int n=0;n<num_kf;n++)
	{
		for (int i = 0; i < 7; i++) {
			kf_sw.statePost.at<float>(i) = states_in_global[i+7*n];
		}
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				kf_sw.errorCovPost.at<float>(i, j) = covariances_in_global[j + i*7 + 7*7*n];
			}
		}
		
		kf_sw.predict();
		
		for (int i = 0; i < 7; i++) {
			 states_out_global[i+7*n] = kf_sw.statePre.at<float>(i);
		}
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				covariances_out_global[j + i*7 + 7*7*n] = kf_sw.errorCovPre.at<float>(i, j) ;
			}
		}
	}
}

void sw_correct(cv::KalmanFilter kf_sw, size_t num_kf) 
{
	for(int n=0;n<num_kf;n++)
	{
		cv::Mat meas(4, 1, CV_32FC1);
		for (int i = 0; i < 4; i++) {
			meas.at<float>(i) = measurements_global[i+4*n];
		}
		
		for (int i = 0; i < 7; i++) {
			kf_sw.statePre.at<float>(i) = states_in_global[i+7*n];
		}
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				kf_sw.errorCovPre.at<float>(i, j) = covariances_in_global[j + i*7 + 7*7*n];
			}
		}

		kf_sw.correct(meas);
		
		for (int i = 0; i < 7; i++) {
			 states_out_global[i+7*n] = kf_sw.statePost.at<float>(i);
		}
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				covariances_out_global[j + i*7 + 7*7*n] = kf_sw.errorCovPost.at<float>(i, j) ;
			}
		}
	}
}

void predict(KFMOT kf_fpga, cv::KalmanFilter kf_sw, std::vector<KalmanTracker>& trackers) {

	for (size_t idx = 0; idx < trackers.size(); idx++) {
		auto &tracker = trackers[idx];
		memcpy(states_in_global+(idx*7), tracker.getState().data, 7*sizeof(float));
		
		memcpy(covariances_in_global+(idx*7*7), tracker.getErrorCov().data,
			   7*7*sizeof(float));
			   
		memcpy(covariancesD_in_global+(idx*7), tracker.getErrorCovD().data,
			   7*sizeof(float));
			   
		tracker.m_age += 1;
		if (tracker.m_time_since_update > 0)
			tracker.m_hit_streak = 0;
		tracker.m_time_since_update += 1;
	}
	
	if(hw_kf_en==1)
		 kf_fpga.kalmanfilter_predict( (int)trackers.size(), states_in_global, covariances_in_global, covariancesD_in_global,
		         states_out_global, covariances_out_global, covariancesD_out_global);
	else
		sw_predict(kf_sw, trackers.size());

	unsigned idx = 0;
	for (auto it = trackers.begin(); it != trackers.end();) {

	   	it->setState(states_out_global+idx*7);
		auto Rect = it->getRect();
		if (Rect.x >= 0 && Rect.y >= 0) {	
			it->setErrorCov(covariances_out_global+idx*7*7);
			it->setErrorCovD(covariancesD_out_global+idx*7);
			it++;
		}
		// Else remove tracker
		else {
			it = trackers.erase(it);
		}
		idx++;
	}
}

void update(KFMOT kf_fpga, cv::KalmanFilter kf_sw, vector<KalmanTracker> &trackers, vector<TrackingBox> &detections,
			vector<cv::Point>& matchedPairs) {


	for (size_t idx = 0; idx < matchedPairs.size(); idx++) {
		auto &tracker = trackers[matchedPairs[idx].x];
		auto &detection = detections[matchedPairs[idx].y];

		memcpy(states_in_global+(idx*7), tracker.getState().data, 7*sizeof(float));
		memcpy(covariances_in_global+(idx*7*7), tracker.getErrorCov().data,
			   7*7*sizeof(float));
		memcpy(covariancesD_in_global+(idx*7), tracker.getErrorCovD().data,
			  7*sizeof(float));
		
		*(measurements_global+(idx*4)) = detection.box.x + detection.box.width / 2;
		*(measurements_global+(idx*4)+1) = detection.box.y + detection.box.height / 2;
		*(measurements_global+(idx*4)+2) = detection.box.area();
		*(measurements_global+(idx*4)+3) = detection.box.width / detection.box.height;
		
		tracker.m_time_since_update = 0;
		tracker.m_hits += 1;
		tracker.m_hit_streak += 1;
	}

	if(hw_kf_en==1)
		kf_fpga.kalmanfilter_correct(matchedPairs.size(), states_in_global, covariances_in_global, covariancesD_in_global, measurements_global,
		states_out_global, covariances_out_global, covariancesD_out_global);
	else
		sw_correct(kf_sw, matchedPairs.size());

	int idx = 0;
    for (auto matchedPair : matchedPairs) {
      	auto &tracker = trackers[matchedPair.x];
		tracker.setState(states_out_global+idx*7);
		tracker.setErrorCov(covariances_out_global+idx*7*7);
		tracker.setErrorCovD(covariancesD_out_global+idx*7);
    	idx++;
    }
}

void TestSORT(string inputDir, bool display)
{
	cout << "Processing " << inputDir << "..." << endl;

	// 1. read detection file
	ifstream detectionFile;
	string detFileName = inputDir + "/det/det.txt";
	detectionFile.open(detFileName);

	if (!detectionFile.is_open())
	{
		cerr << "Error: can not find file " << detFileName << endl;
		return;
	}

	string detLine;
	istringstream ss;
	vector<TrackingBox> detData;
	char ch;
	float tpx, tpy, tpw, tph;

	while ( getline(detectionFile, detLine) )
	{
		TrackingBox tb;

		ss.str(detLine);
		ss >> tb.frame >> ch >> tb.id >> ch;
		ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph;
		ss.str("");

		tb.box = Rect_<float>(Point_<float>(tpx, tpy), Point_<float>(tpx + tpw, tpy + tph));
		detData.push_back(tb);
	}
	detectionFile.close();

	// 2. group detData by frame
	int maxFrame = 0;
	for (auto tb : detData) // find max frame number
	{
		if (maxFrame < tb.frame)
			maxFrame = tb.frame;
	}

	vector<vector<TrackingBox>> detFrameData;
	vector<TrackingBox> tempVec;
	for (int fi = 0; fi < maxFrame; fi++)
	{
		for (auto tb : detData)
			if (tb.frame == fi + 1) // frame num starts from 1
				tempVec.push_back(tb);
		detFrameData.push_back(tempVec);
		tempVec.clear();
	}

	// 3. update across frames
	int frame_count = 0;
	int max_age = 30;
	int min_hits = 3;
	double iouThreshold = 0.7;
	vector<KalmanTracker> trackers;
	KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

	// variables used in the for-loop
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allDetections;
	set<int> matchedDetections;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	double cycle_time = 0.0;
	int64 start_time = 0;

	// prepare result file.
	ofstream resultsFile;
	string resFileName = "output.txt";
	resultsFile.open(resFileName);

	if (!resultsFile.is_open())
	{
		cerr << "Error: can not create file " << resFileName << endl;
		return;
	}

	KFMOT kf_fpga;
	cv::KalmanFilter kf_sw;

	if(hw_kf_en==1)
	{
	    kf_fpga.kalmanfilter_init("/media/sd-mmcblk0p1/krnl_kalmanfilter.xclbin",
				"kalmanfilter_accel",
				0);
	}
	else
	{
		int stateNum = 7;
		int measureNum = 4;
		kf_sw = KalmanFilter(stateNum, measureNum, 0);
	
		kf_sw.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
			1, 0, 0, 0, 1, 0, 0,
			0, 1, 0, 0, 0, 1, 0,
			0, 0, 1, 0, 0, 0, 1,
			0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 1);
		setIdentity(kf_sw.measurementMatrix);
		setIdentity(kf_sw.processNoiseCov, Scalar::all(1e-2));
		setIdentity(kf_sw.measurementNoiseCov, Scalar::all(1e-1));
	
	}

	//////////////////////////////////////////////
	// main loop
	for (int fi = 0; fi < maxFrame; fi++)
	{
		total_frames++;
		frame_count++;

		start_time = getTickCount();

		if (trackers.size() == 0) // the first frame met
		{
			trackers.reserve(detFrameData[fi].size());
			// initialize kalman trackers using first detections.
			// __TIC_SUM__(INITIALISE);
			for (unsigned int i = 0; i < detFrameData[fi].size(); i++)
			{
				trackers.emplace_back(detFrameData[fi][i].box);
			}
			// __TOC_SUM__(INITIALISE);
			// output the first frame detections
			for (unsigned int id = 0; id < detFrameData[fi].size(); id++)
			{
				TrackingBox tb = detFrameData[fi][id];
				resultsFile << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
			}
			continue;
		}

		///////////////////////////////////////
		// 3.1. get predicted locations from existing trackers.
		// __TIC_SUM__(PREDICT);
		predict(kf_fpga, kf_sw, trackers);
		// __TOC_SUM__(PREDICT);

		///////////////////////////////////////
		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		// dets : detFrameData[fi]
		trkNum = trackers.size();
		detNum = detFrameData[fi].size();

		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		// __TIC_SUM__(IOU);
		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			auto rect = trackers[i].getRect();
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(rect, detFrameData[fi][j].box);
			}
		}
		// __TOC_SUM__(IOU);

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();

		// __TIC_SUM__(HUNGARIAN);
		HungAlgo.Solve(iouMatrix, assignment);
		// __TOC_SUM__(HUNGARIAN);

		// find matches, unmatched_detections and unmatched_predictions
		// filter out matched with low IOU

		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allDetections.clear();
		matchedDetections.clear();
		matchedPairs.clear();

		// __TIC_SUM__(IOU_THRESHOLD);
		for (unsigned int n = 0; n < detNum; n++)
			allDetections.insert(n);

		for (unsigned int i = 0; i < trkNum; ++i) {
			if (assignment[i] == -1)
				unmatchedTrajectories.insert(i);
			else if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else {
				matchedDetections.insert(assignment[i]);
				matchedPairs.emplace_back(i, assignment[i]);
			}
		}

		set_difference(allDetections.begin(), allDetections.end(),
			matchedDetections.begin(), matchedDetections.end(),
			insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		// __TOC_SUM__(IOU_THRESHOLD);

		///////////////////////////////////////
		// 3.3. updating trackers

		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		// __TIC_SUM__(UPDATE);
		
		int detIdx2, trkIdx2;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx2 = matchedPairs[i].x;
			detIdx2 = matchedPairs[i].y;
		}
		
		update(kf_fpga, kf_sw, trackers, detFrameData[fi], matchedPairs);
		// __TOC_SUM__(UPDATE);

		// create and initialise new trackers for unmatched detections
		// __TIC_SUM__(INITIALISE_NEW);
		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(detFrameData[fi][umd].box);
			trackers.push_back(tracker);
		}
		// __TOC_SUM__(INITIALISE_NEW);

		// get trackers' output
		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
			{
				TrackingBox res;
				res.box = (*it).getRect();
				res.id = (*it).m_id + 1;
				res.frame = frame_count;
				frameTrackingResult.push_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}

		cycle_time = (double)(getTickCount() - start_time);
		total_time += cycle_time / getTickFrequency();

		for (auto tb : frameTrackingResult)
			resultsFile << tb.frame << "," << tb.id << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
	}

	resultsFile.close();
}
