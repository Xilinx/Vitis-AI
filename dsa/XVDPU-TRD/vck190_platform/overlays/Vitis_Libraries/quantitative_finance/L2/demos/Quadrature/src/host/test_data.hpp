#ifndef _TEST_DATA_H_
#define _TEST_DATA_H_

#define TEST_DT float
// generated from test_vectors/test_vectors_heston.txt using model HestonQL on Tue Oct  1 15:07:57 2019
struct test_data_type {
    TEST_DT s;
    TEST_DT k;
    TEST_DT t;
    TEST_DT v;
    TEST_DT r;
    TEST_DT rho;
    TEST_DT vvol;
    TEST_DT vbar;
    TEST_DT kappa;
    TEST_DT exp;
};

struct test_data_type test_data[] = {
    {80, 100, 0.5, 0.01, 0.05, -0.5, 0.1, 0.5, 1, 2.308462},
};

#endif /* include only once */
