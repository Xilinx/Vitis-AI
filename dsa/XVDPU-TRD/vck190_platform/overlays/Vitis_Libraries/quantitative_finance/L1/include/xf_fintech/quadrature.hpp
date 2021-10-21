/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @brief quadrature.hpp
 * @brief This file includes numerical integration methods
 */

#ifndef _XF_QUADRATURE_H_
#define _XF_QUADRATURE_H_

namespace xf {
namespace fintech {

// stack start ----------------------------
// internal functions for implementing a simple stack
template <typename DT>
struct stack_type {
    DT stack[MAX_DEPTH];
    int index;
};

template <typename DT>
static void stack_init(struct stack_type<DT>* s) {
    s->index = 0;
}

template <typename DT>
static int stack_push(struct stack_type<DT>* s, DT value) {
    if (s->index >= MAX_DEPTH) {
        /* no more room on stack */
        return 0;
    }
    s->stack[s->index] = value;
    (s->index)++;
    return 1;
}

template <typename DT>
static int stack_pop(struct stack_type<DT>* s, DT* value) {
    if (s->index == 0) {
        /* stack empty */
        return 0;
    }
    (s->index)--;
    *value = s->stack[s->index];
    return 1;
}

// stack end -------------------------------

/**
 * @brief  integration function using the adaptive trapezoidal technique
 *
 * @tparam DT   float or DT, determines the precision of the result
 *
 * @param  a    lower limit of integration
 * @param  b    upper limit of integration
 * @param  tol  required tolerance of the result
 * @param  res  the result of the integration
 * @param  p    pointer to structure containing parameters for integrating function
 * @return      1 - success, 0 - fail (due to stack limitation)
 */
template <typename DT>
static int trap_integrate(DT a, DT b, DT tol, DT* res, XF_USER_DATA_TYPE* p) {
    int still_processing = 1;
    DT orig_tol = tol;
    DT orig_a = a;
    DT orig_b = b;
    int cnt = 0;
    int ret = 1;

    struct stack_type<DT> b_stack;
    struct stack_type<DT> fb_stack;
    stack_init(&b_stack);
    stack_init(&fb_stack);

    (*res) = 0;
    DT fa = XF_INTEGRAND_FN(a, p);
    DT fb = XF_INTEGRAND_FN(b, p);
    while (still_processing && cnt < MAX_ITERATIONS) {
        cnt++;
        DT h = b - a;
        DT mid = (a + b) / 2;
        DT fmid = XF_INTEGRAND_FN(mid, p);
        DT area1 = h * (fa + fb) / 2;
        DT area2 = (h / 2 * (fa + fmid) / 2) + (h / 2 * (fmid + fb) / 2);

        tol = orig_tol * (b - a) / (orig_b - orig_a);

        if (fabs(area1 - area2) < (3 * tol)) {
            (*res) += area2;
            a = b;
            fa = fb;
            if (!stack_pop(&b_stack, &b)) {
                still_processing = 0;
            }
            stack_pop(&fb_stack, &fb); /* assume if we can pop b we can pop fb */
        } else {
            if (!stack_push(&b_stack, b)) {
                still_processing = 0;
                ret = 0; /* indicate integration failed (due to blowing stack) */
            }
            stack_push(&fb_stack, fb); /* assume if we can push b we can push fb */
            b = mid;
            fb = fmid;
            /* a and fa stay the same */
        }
    }
    return ret;
}

/*
 * internal function to calculates simpson approximation for a segment
 */
template <typename DT>
static DT simp_elem(DT a, DT b, XF_USER_DATA_TYPE* p) {
    DT h = (b - a) / 2;
    DT mid = (a + b) / 2;
    return h / 3 * (XF_INTEGRAND_FN(a, p) + 4 * XF_INTEGRAND_FN(mid, p) + XF_INTEGRAND_FN(b, p));
}

/**
 * @brief  integration function using the adaptive simpson 1/3 technique
 *
 * @tparam DT   float or DT, determines the precision of the result
 *
 * @param  a    lower limit of integration
 * @param  b    upper limit of integration
 * @param  tol  required tolerance of the result
 * @param  res  the result of the integration
 * @param  p    pointer to structure containing parameters for integrating function
 * @return      1 - success, 0 - fail (due to stack limitation)
 */
template <typename DT>
static int simp_integrate(DT a, DT b, DT tol, DT* res, XF_USER_DATA_TYPE* p) {
    int still_processing = 1;
    DT orig_tol = tol;
    DT orig_a = a;
    DT orig_b = b;
    int cnt = 0;
    int ret = 1;

    struct stack_type<DT> b_stack;
    struct stack_type<DT> d_stack;
    struct stack_type<DT> fb_stack;
    struct stack_type<DT> fd_stack;
    stack_init(&b_stack);
    stack_init(&d_stack);
    stack_init(&fb_stack);
    stack_init(&fd_stack);

    (*res) = 0;
    DT fa = XF_INTEGRAND_FN(a, p);
    DT fb = XF_INTEGRAND_FN(b, p);
    DT mid = (a + b) / 2;
    DT fmid = XF_INTEGRAND_FN(mid, p);
    while (still_processing && cnt < MAX_ITERATIONS) {
        cnt++;
        DT m = (b - a) / 6;
        DT area1 = m * (fa + 4 * fmid + fb);

        DT c = (a + mid) / 2;
        DT fc = XF_INTEGRAND_FN(c, p);
        m /= 2;
        DT left = m * (fa + 4 * fc + fmid);

        DT d = (mid + b) / 2;
        DT fd = XF_INTEGRAND_FN(d, p);
        DT right = m * (fmid + 4 * fd + fb);

        DT area2 = left + right;

        tol = orig_tol * (b - a) / (orig_b - orig_a);

        if (fabs(area1 - area2) < (15 * tol)) {
            (*res) += area2;
            a = b;
            fa = fb;
            if (!stack_pop(&b_stack, &b)) {
                still_processing = 0;
            }
            /* if we can pop b, assume we can pop the rest */
            stack_pop(&fb_stack, &fb);
            stack_pop(&d_stack, &mid);
            stack_pop(&fd_stack, &fmid);
        } else {
            if (!stack_push(&b_stack, b)) {
                still_processing = 0;
                ret = 0; /* indicate integration failed (due to blowing stack) */
            }
            /* if we can push b, assume we can push the rest */
            stack_push(&d_stack, d);
            stack_push(&fb_stack, fb);
            stack_push(&fd_stack, fd);
            b = mid;
            fb = fmid;
            mid = c;
            fmid = fc;
            /* a and fa stay the same */
        }
    }
    return ret;
}

/**
 * @brief  integration function using the romberg technique
 * Based on https://en.wikipedia.org/wiki/Romberg%27s_method
 *
 * @tparam DT   float or DT, determines the precision of the result
 *
 * @param  a    lower limit of integration
 * @param  b    upper limit of integration
 * @param  tol  required tolerance of the result
 * @param  res  the result of the integration
 * @param  p    pointer to structure containing parameters for integrating function
 * @return      1 - success, 0 - fail (due to stack limitation)
 */
template <typename DT>
static int romberg_integrate(DT a, DT b, DT tol, DT* res, XF_USER_DATA_TYPE* p) {
    DT R1[MAX_ITERATIONS];
    DT R2[MAX_ITERATIONS];
    DT *Rp = &R1[0], *Rc = &R2[0];                                    // Rp is previous row, Rc is current row
    DT h = (b - a);                                                   // step size
    Rp[0] = (XF_INTEGRAND_FN(a, p) + XF_INTEGRAND_FN(b, p)) * h * .5; // first trapezoidal step

    (*res) = 0;
    int i = 0;
    for (i = 1; i < MAX_ITERATIONS; ++i) {
        h /= 2.;
        DT c = 0;
        int ep = 1 << (i - 1); // 2^(n-1)
        for (int j = 1; j <= ep; ++j) {
            c += XF_INTEGRAND_FN(a + (2 * j - 1) * h, p);
        }
        Rc[0] = h * c + .5 * Rp[0]; // R(i,0)

        for (int j = 1; j <= i; ++j) {
            DT n_k = pow(4, j);
            Rc[j] = (n_k * Rc[j - 1] - Rp[j - 1]) / (n_k - 1); // compute R(i,j)
        }

        if (i > 1 && fabs(Rp[i - 1] - Rc[i]) < tol) {
            /* found result within max steps - good result */
            *res = Rc[i - 1];
            return 1;
        }

        // swap Rn and Rc as we only need the last row
        DT* rt = Rp;
        Rp = Rc;
        Rc = rt;
    }

    /* used up max steps, this is our best guess */
    *res = Rp[MAX_ITERATIONS - 1];
    return 0;
}

} // namespace fintech
} // namespace xf

#endif /* _XF_QUADRATURE_H_ */
