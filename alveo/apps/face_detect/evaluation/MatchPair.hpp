#ifndef __MATCHPAIR_HPP__
#define __MATCHPAIR_HPP__

#include "Region.hpp"

/**
 * Specifies a pair for two regions as matched regions
 * */
class MatchPair{
    public:
	// First region in the pair
	Region *r1;
	// Second region in the pair
	Region *r2;
	// Match score
	double score;
	// Constructor
	MatchPair(Region *a, Region *b, double s);
	// Destructor
	~MatchPair();
};

#endif
