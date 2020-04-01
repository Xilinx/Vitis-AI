#ifndef __MATCHING_HPP__
#define __MATCHING_HPP__

#include <string>
#include <vector>
#include <iostream>

#include "Region.hpp"
#include "RegionsSingleImage.hpp"
#include "MatchPair.hpp"
#include "Results.hpp"
#include "Hungarian.hpp"

using std::vector;

/**
 * Class that computes the matching between annotated and 
 * detected regions
 * */
class Matching{
    public:
	/// Name of the algorithm used for matching
	std::string matchingAlgoStr;
	/// Constructor
	Matching(RegionsSingleImage *, RegionsSingleImage *);
	/// Constructor
	Matching(std::string, RegionsSingleImage *, RegionsSingleImage *);
	/// Destructor
	~Matching();
	/// Compute the matching pairs.
	/// Returns a vector of MatchPair pointers
	vector<MatchPair *> * getMatchPairs();

    private:
	/// Set of annotated regions
	RegionsSingleImage *annot;
	/// Set of detected regions
	RegionsSingleImage *det;
	/// Matrix of matching scores for annnotation and detections
	vector<vector<double> *> * pairWiseScores;

	/// Computes the score for a single pair of regions
	double computeScore(Region *, Region *);
	/// populate the pairWiseScores matrix
	void computePairWiseScores();
	/// Free the memory for the pairWiseScores matrix
	void clearPairWiseScores();
	/// Runs the Hungarian algorithm
	vector<MatchPair *> * runHungarian();
};

#endif
