#include "MatchPair.hpp"

MatchPair::MatchPair(Region *a, Region *b, double s)
{
    r1=a; r2=b; score=s;
}

MatchPair::~MatchPair()
{
}

