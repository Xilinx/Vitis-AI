#ifndef __RESULTS_HPP__
#define __RESULTS_HPP__

#include<vector>
#include<string>
#include <fstream>
#include "MatchPair.hpp"
#include "RegionsSingleImage.hpp"

using std::vector;
using std::string;

/**
 * Specifies the cumulative result statistics
 * */
class Results{
    private:
	/// number of annotated regions (TP+FN)
	unsigned int N;
	/// threshold used for computing this result
	double scoreThreshold;
	/// True positives -- continuous
	double TPCont;
	/// True positives -- discrete
	double TPDisc;
	/// False positives -- discrete
	double FP;
	/// Name of the image 
	string imName;

    public:
	/// Constructor
	Results();
	/// Constructor -- copy the contents of *r
	Results(Results *r);
	/// Constructor -- copy the contents of *r and add N2 to N
	Results(Results *r, int N2);
	/// Constructor -- merge r1 and r2
	Results(Results *r1, Results *r2);
	/// Constructor
	Results(string imName, double threshold, vector<MatchPair *> *matchPairs, RegionsSingleImage *annot, RegionsSingleImage *det);
	/// Return a vector of results with combined statistics from the two 
	/// vectors rv1 and rv2
	vector<Results *> * merge(vector<Results *> *rv1, vector<Results *> *rv2);
	/// print this result into the ostream os
	void print(std::ostream &os);
	/// save the ROC curve computed from rv into the file outFile
	void saveROC(string outFile, vector<Results *> *rv);

	/// get N
	unsigned int getN();
};

#endif
