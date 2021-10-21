
#ifndef __XF_GRAPH_ENUMS_HPP_
#define __XF_GRAPH_ENUMS_HPP_

#include <ap_int.h>

namespace xf {

namespace graph {

namespace enums {
enum SimilarityType {
    JACCARD_SIMILARITY = 0,
    COSINE_SIMILARITY = 1,
};

enum GraphType { SPARSE = 0, DENSE = 1 };

enum DataType { UINT32_TYPE = 0, FLOAT_TYPE = 1, UINT64_TYPE = 2, DOUBLE_TYPE = 3, INT32_TYPE = 4, INT64_TYPE = 5 };

} // enums

} // graph

} // xf

#endif
