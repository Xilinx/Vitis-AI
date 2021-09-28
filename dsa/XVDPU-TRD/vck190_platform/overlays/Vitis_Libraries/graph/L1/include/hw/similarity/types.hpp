
#ifndef __XF_GRAPH_TYPES_HPP_
#define __XF_GRAPH_TYPES_HPP_

#include <ap_int.h>

template <typename I, typename F>
inline F bitsToFloat(I in) {
    union {
        I __I;
        F __F;
    } __T;
    __T.__I = in;
    return __T.__F;
}

template <typename F, typename I>
inline I floatToBits(F in) {
    union {
        I __I;
        F __F;
    } __T;
    __T.__F = in;
    return __T.__I;
}

template <typename T>
struct isFloat {
    operator bool() { return false; }
};

template <>
struct isFloat<float> {
    operator bool() { return true; }
};

#endif
