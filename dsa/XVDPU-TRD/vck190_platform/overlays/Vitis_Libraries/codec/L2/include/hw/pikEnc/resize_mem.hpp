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
 * @file resize_mem.hpp
 */

#ifndef _XF_CODEC_RESIZE_MEM_HPP_
#define _XF_CODEC_RESIZE_MEM_HPP_

#include "ap_int.h"
#include <assert.h>

typedef ap_uint<32> HLS_SIZE_T;
typedef ap_uint<5> HLS_CHANNEL_T;

/* Template class of Window */
template <int ROWS, int COLS, typename T>
class Window {
   public:
    Window(){
#pragma HLS ARRAY_PARTITION variable = val dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = val dim = 2 complete
    };

    /* Window main APIs */
    void shift_pixels_left();
    void shift_pixels_right();
    void shift_pixels_up();
    void shift_pixels_down();
    void shift_diagonal();
    void insert_pixel(T value, int row, int col);
    void insert_row(T value[COLS], int row);
    void insert_top_row(T value[COLS]);
    void insert_bottom_row(T value[COLS]);
    void insert_col(T value[ROWS], int col);
    void insert_left_col(T value[ROWS]);
    void insert_right_col(T value[ROWS]);
    void copy_one_row(int row1, int row2);
    void copy_one_col(int col1, int col2);

    T& getval(int row, int col);
    T& operator()(int row, int col);

    /* Back compatible APIs */
    void shift_left();
    void shift_right();
    void shift_up();
    void shift_down();
    void insert(T value, int row, int col);
    void insert_top(T value[COLS]);
    void insert_bottom(T value[COLS]);
    void insert_left(T value[ROWS]);
    void insert_right(T value[ROWS]);
    // T& getval(int row, int col);
    // T& operator ()(int row, int col);

    T val[ROWS][COLS];
#ifndef __SYNTHESIS__
    void restore_val();
    void window_print();
    T val_t[ROWS][COLS];
#endif
};

/* Member functions of Window class */
/* Origin in upper-left point */
/*       0   1        C-2 C-1
 *     +---+---+-...-+---+---+
 *  0  |   |   |     |   |   |
 *     +---+---+-...-+---+---+
 *  1  |   |   |     |   |   |
 *     +---+---+-...-+---+---+
 *       ...     ...    ...
 *     +---+---+-...-+---+---+
 * R-2 |   |   |     |   |   |
 *     +---+---+-...-+---+---+
 * R-1 |   |   |     |   |   |
 *     +---+---+-...-+---+---+
 *
 */

/*
 * Window content shift left
 * Assumes new values will be placed in right column = COLS-1
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::shift_pixels_left() {
#pragma HLS inline

    HLS_SIZE_T i, j;
    for (i = 0; i < ROWS; i++) {
#pragma HLS unroll
        for (j = 0; j < COLS - 1; j++) {
#pragma HLS unroll
            val[i][j] = val[i][j + 1];
        }
    }
}

/*
 * Window content shift right
 * Assumes new values will be placed in left column = 0
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::shift_pixels_right() {
#pragma HLS inline

    HLS_SIZE_T i, j;
    for (i = 0; i < ROWS; i++) {
#pragma HLS unroll
        for (j = COLS - 1; j > 0; j--) {
#pragma HLS unroll
            val[i][j] = val[i][j - 1];
        }
    }
}

/*
 * Window content shift up
 * Assumes new values will be placed in bottom row = ROWS-1
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::shift_pixels_up() {
#pragma HLS inline

    HLS_SIZE_T i, j;
    for (i = 0; i < ROWS - 1; i++) {
#pragma HLS unroll
        for (j = 0; j < COLS; j++) {
#pragma HLS unroll
            val[i][j] = val[i + 1][j];
        }
    }
}

template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::copy_one_col(int col1, int col2) {
#pragma HLS inline
    assert(col1 >= 0 && col1 < COLS && col2 >= 0 && col2 < COLS);
    HLS_SIZE_T i;
    for (i = 0; i < ROWS; i++) {
#pragma HLS unroll
        val[i][col1] = val[i][col2];
    }
}

template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::copy_one_row(int row1, int row2) {
#pragma HLS inline
    assert(row1 >= 0 && row1 < ROWS && row2 >= 0 && row2 < ROWS);
    HLS_SIZE_T j;
    for (j = 0; j < COLS; j++) {
        val[row1][j] = val[row2][j];
    }
}

/*
 * Window content shift down
 * Assumes new values will be placed in top row = 0
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::shift_pixels_down() {
#pragma HLS inline

    HLS_SIZE_T i, j;
    for (i = ROWS - 1; i > 0; i--) {
#pragma HLS unroll
        for (j = 0; j < COLS; j++) {
#pragma HLS unroll
            val[i][j] = val[i - 1][j];
        }
    }
}

template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::shift_diagonal() {
#pragma HLS inline off
    assert(ROWS == COLS);
    HLS_SIZE_T i, j;

    T tmp;
    for (i = 0; i < ROWS; i++) {
#pragma HLS unroll
        for (j = 0; j < COLS; j++) {
#pragma HLS unroll
            if (i < j) {
                tmp = val[j][i];
                val[j][i] = val[i][j];
                val[i][j] = tmp;
            }
        }
    }
}

/* Window insert pixel
 * Inserts a new value at any location of the window
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_pixel(T value, int row, int col) {
#pragma HLS inline
    assert(row >= 0 && row < ROWS && col >= 0 && col < COLS);

    val[row][col] = value;
}

/* Window insert row
 * Inserts a set of values in any row of the window
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_row(T value[COLS], int row) {
#pragma HLS inline

    HLS_SIZE_T j;
    for (j = 0; j < COLS; j++) {
#pragma HLS unroll
        val[row][j] = value[j];
    }
}

/* Window insert top row
 * Inserts a set of values in top row = 0 of the window
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_top_row(T value[COLS]) {
#pragma HLS inline

    insert_row(value, 0);
}

/* Window insert bottom row
 * Inserts a set of values in bottom row = ROWS-1 of the window
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_bottom_row(T value[COLS]) {
#pragma HLS inline

    insert_row(value, ROWS - 1);
}

/* Window insert column
 * Inserts a set of values in any column of the window
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_col(T value[ROWS], int col) {
#pragma HLS inline

    HLS_SIZE_T i;
    for (i = 0; i < ROWS; i++) {
#pragma HLS unroll
        val[i][col] = value[i];
    }
}

/* Window insert left column
 * Inserts a set of values in left column = 0 of the window
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_left_col(T value[ROWS]) {
#pragma HLS inline

    insert_col(value, 0);
}

/* Window insert right column
 * Inserts a set of values in right column = COLS-1 of the window
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_right_col(T value[ROWS]) {
#pragma HLS inline

    insert_col(value, COLS - 1);
}

/* Window getval
 * Returns the data value in the window at position (row,col)
 */
template <int ROWS, int COLS, typename T>
T& Window<ROWS, COLS, T>::getval(int row, int col) {
#pragma HLS inline
    assert(row >= 0 && row < ROWS && col >= 0 && col < COLS);
    return val[row][col];
}

/* Window getval
 * Returns the data value in the window at position (row,col)
 */
template <int ROWS, int COLS, typename T>
T& Window<ROWS, COLS, T>::operator()(int row, int col) {
#pragma HLS inline
    return getval(row, col);
}

#ifndef __SYNTHESIS__
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::restore_val() {
    HLS_SIZE_T i, j;
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            val_t[i][j] = val[i][j];
        }
    }
}

template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::window_print() {
    HLS_SIZE_T i, j;
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            std::cout << std::setw(5) << val[i][j];
        }
    }
    std::cout << "\n";
}
#endif

/* NOTE:
 * Back compatible APIs, take bottom-right point as the origin
 * Window shift left, while contents shift right
 * Assumes new values will be placed in left column(=COLS-1)
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::shift_left() {
#pragma HLS inline
    shift_pixels_left(); // take upper-left point as origin
}

/* NOTE:
 * Back compatible APIs, take bottom-right point as the origin
 * Window shift right, while contents shift left
 * Assumes new values will be placed in right column(=0)
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::shift_right() {
#pragma HLS inline
    shift_pixels_right(); // take upper-left point as origin
}

/* NOTE:
 * Back compatible APIs, take bottom-right point as the origin
 * Window shift up, while contents shift down
 * Assumes new values will be placed in top row(=ROWS-1)
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::shift_up() {
#pragma HLS inline
    shift_pixels_up(); // take upper-left point as origin
}

/* NOTE:
 * Back compatible APIs, take bottom-right point as the origin
 * Window shift down, while contents shift up
 * Assumes new values will be placed in bottom row(=0)
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::shift_down() {
#pragma HLS inline
    shift_pixels_down(); // take upper-left point as origin
}

/* NOTE:
 * Back compatible APIs, take bottom-right point as the origin
 * Window insert
 * Inserts a new value at any location of the window
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert(T value, int row, int col) {
#pragma HLS inline
    insert_pixel(value, row, col);
}

/* NOTE:
 * Back compatible APIs, take bottom-right point as the origin
 * Window insert top
 * Inserts a set of values in top row(=ROWS-1)
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_top(T value[COLS]) {
#pragma HLS inline
    insert_bottom_row(value);
}

/* NOTE:
 * Back compatible APIs, take bottom-right point as the origin
 * Window insert bottom
 * Inserts a set of values in bottom row(=0)
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_bottom(T value[COLS]) {
#pragma HLS inline
    insert_top_row(value);
}

/* NOTE:
 * Back compatible APIs, take bottom-right point as the origin
 * Window insert left
 * Inserts a set of values in left column(=COLS-1)
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_left(T value[ROWS]) {
#pragma HLS inline
    insert_right_col(value);
}

/* NOTE:
 * Back compatible APIs, take bottom-right point as the origin
 * Window insert right
 * Inserts a set of values in right column(=0)
 */
template <int ROWS, int COLS, typename T>
void Window<ROWS, COLS, T>::insert_right(T value[ROWS]) {
#pragma HLS inline
    insert_left_col(value);
}

/* Template class of Line Buffer */
template <int ROWS, int COLS, typename T, int RESHAPE = 0>
class LineBuffer;

template <int ROWS, int COLS, typename T>
class LineBuffer<ROWS, COLS, T, 0> {
   public:
    LineBuffer(){
#pragma HLS array_partition variable = val dim = 1 complete
#pragma HLS dependence variable = val inter false
#pragma HLS dependence variable = val intra false
    };
    /* LineBuffer main APIs */
    void shift_pixels_up(int col);
    void shift_pixels_down(int col);
    void insert_bottom_row(T value, int col);
    void insert_top_row(T value, int col);
    void get_col(T value[ROWS], int col);
    T& getval(int row, int col);
    T& operator()(int row, int col);

    /* Back compatible APIs */
    void shift_up(int col);
    void shift_down(int col);
    void insert_bottom(T value, int col);
    void insert_top(T value, int col);
    // T& getval(int row, int col);
    // T& operator ()(int row, int col);

    T val[ROWS][COLS];
};

/* Member functions of LineBuffer class */
/* Origin in upper-left point */
/*       0   1            C-2 C-1
 *     +---+---+-... ...-+---+---+
 *  0  |   |   |         |   |   |
 *     +---+---+-... ...-+---+---+
 *  1  |   |   |         |   |   |
 *     +---+---+-... ...-+---+---+
 *       ...     ... ...    ...
 *     +---+---+-... ...-+---+---+
 * R-2 |   |   |         |   |   |
 *     +---+---+-... ...-+---+---+
 * R-1 |   |   |         |   |   |
 *     +---+---+-... ...-+---+---+
 *
 */

/* Member functions of LineBuffer class */

/*
 * LineBuffer content shift down
 * Assumes new values will be placed in top row = 0
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T>::shift_pixels_down(int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);
    HLS_SIZE_T i;
    for (i = ROWS - 1; i > 0; i--) {
#pragma HLS unroll
        val[i][col] = val[i - 1][col];
    }
}

/*
 * LineBuffer content shift up
 * Assumes new values will be placed in top row = ROWS-1
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T>::shift_pixels_up(int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);

    HLS_SIZE_T i;
    for (i = 0; i < ROWS - 1; i++) {
#pragma HLS unroll
        val[i][col] = val[i + 1][col];
    }
}

/* LineBuffer insert bottom row
 * Inserts a new value in bottom row= ROWS-1 of the linebuffer
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T>::insert_bottom_row(T value, int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);

    val[ROWS - 1][col] = value;
}

/* LineBuffer insert top row
 * Inserts a new value in top row=0 of the linebuffer
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T>::insert_top_row(T value, int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);

    val[0][col] = value;
}

/* LineBuffer get a column
 * Get a column value of the linebuffer
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T>::get_col(T value[ROWS], int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);
    HLS_SIZE_T i;
    for (i = 0; i < ROWS; i++) {
#pragma HLS unroll
        value[i] = val[i][col];
    }
}

/* Line buffer getval
 * Returns the data value in the line buffer at position row, col
 */
template <int ROWS, int COLS, typename T>
T& LineBuffer<ROWS, COLS, T>::getval(int row, int col) {
#pragma HLS inline
    assert(row >= 0 && row < ROWS && col >= 0 && col < COLS);
    return val[row][col];
}

/* Line buffer getval
 * Returns the data value in the line buffer at position row, col
 */
template <int ROWS, int COLS, typename T>
T& LineBuffer<ROWS, COLS, T>::operator()(int row, int col) {
#pragma HLS inline
    return getval(row, col);
}

/* NOTE:
 * Back compatible APIs, take bottom-left point as the origin
 * LineBuffer shift down, while contents shift up
 * Assumes new values will be placed in bottom row(=0)
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T>::shift_down(int col) {
#pragma HLS inline
    shift_pixels_down(col);
}

/* NOTE:
 * Back compatible APIs, take bottom-left point as the origin
 * LineBuffer shift up, while contents shift down
 * Assumes new values will be placed in top row(=ROWS-1)
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T>::shift_up(int col) {
#pragma HLS inline
    shift_pixels_up(col);
}

/* NOTE:
 * Back compatible APIs, take bottom-left point as the origin
 * LineBuffer insert
 * Inserts a new value in bottom row(=0)
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T>::insert_bottom(T value, int col) {
#pragma HLS inline
    insert_top_row(value, col);
}

/* NOTE:
 * Back compatible APIs, take bottom-left point as the origin
 * LineBuffer insert
 * Inserts a new value in top row(=ROWS-1)
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T>::insert_top(T value, int col) {
#pragma HLS inline
    insert_bottom_row(value, col);
}

template <int ROWS, int COLS, typename T>
class LineBuffer<ROWS, COLS, T, 1> {
   public:
    LineBuffer(){
#pragma HLS RESOURCE variable = val core = XPM_MEMORY uram
#pragma HLS array_reshape variable = val dim = 1
#pragma HLS dependence variable = val inter false
#pragma HLS dependence variable = val intra false
    };
    /* LineBuffer main APIs */
    void shift_pixels_up(int col);
    void shift_pixels_down(int col);
    void insert_bottom_row(T value, int col);
    void insert_top_row(T value, int col);
    void get_col(T value[ROWS], int col);
    T& getval(int row, int col);
    T& operator()(int row, int col);

    /* Back compatible APIs */
    void shift_up(int col);
    void shift_down(int col);
    void insert_bottom(T value, int col);
    void insert_top(T value, int col);
    // T& getval(int row, int col);
    // T& operator ()(int row, int col);

    T val[ROWS][COLS];
#ifndef __SYNTHESIS__
    void restore_val();
    void linebuffer_print(int col);
    T val_t[ROWS][COLS];
#endif
};

/* Member functions of LineBuffer_reshape class */

/* Origin in upper-left point */
/*       0   1            C-2 C-1
 *     +---+---+-... ...-+---+---+
 *  0  |   |   |         |   |   |
 *     +---+---+-... ...-+---+---+
 *  1  |   |   |         |   |   |
 *     +---+---+-... ...-+---+---+
 *       ...     ... ...    ...
 *     +---+---+-... ...-+---+---+
 * R-2 |   |   |         |   |   |
 *     +---+---+-... ...-+---+---+
 * R-1 |   |   |         |   |   |
 *     +---+---+-... ...-+---+---+
 *
 */

/* Member functions of LineBuffer_reshape class */

/*
 * LineBuffer content shift down
 * Assumes new values will be placed in top row = 0
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::shift_pixels_down(int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);

    HLS_SIZE_T i;
    for (i = ROWS - 1; i > 0; i--) {
#pragma HLS unroll
        val[i][col] = val[i - 1][col];
    }
}

/*
 * LineBuffer content shift up
 * Assumes new values will be placed in top row = ROWS-1
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::shift_pixels_up(int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);

    HLS_SIZE_T i;
    for (i = 0; i < ROWS - 1; i++) {
#pragma HLS unroll
        val[i][col] = val[i + 1][col];
    }
}

/* LineBuffer insert bottom row
 * Inserts a new value in bottom row= ROWS-1 of the linebuffer
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::insert_bottom_row(T value, int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);

    val[ROWS - 1][col] = value;
}

/* LineBuffer insert top row
 * Inserts a new value in top row=0 of the linebuffer
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::insert_top_row(T value, int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);

    val[0][col] = value;
}

/* LineBuffer get a column
 * Get a column value of the linebuffer
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::get_col(T value[ROWS], int col) {
#pragma HLS inline
    assert(col >= 0 && col < COLS);
    HLS_SIZE_T i;
    for (i = 0; i < ROWS; i++) {
#pragma HLS unroll
        value[i] = val[i][col];
    }
}

/* Line buffer getval
 * Returns the data value in the line buffer at position row, col
 */
template <int ROWS, int COLS, typename T>
T& LineBuffer<ROWS, COLS, T, 1>::getval(int row, int col) {
#pragma HLS inline
    assert(row >= 0 && row < ROWS && col >= 0 && col < COLS);
    return val[row][col];
}

/* Line buffer getval
 * Returns the data value in the line buffer at position row, col
 */
template <int ROWS, int COLS, typename T>
T& LineBuffer<ROWS, COLS, T, 1>::operator()(int row, int col) {
#pragma HLS inline
    return getval(row, col);
}

/* NOTE:
 * Back compatible APIs, take bottom-left point as the origin
 * LineBuffer shift down, while contents shift up
 * Assumes new values will be placed in bottom row(=0)
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::shift_down(int col) {
#pragma HLS inline
    shift_pixels_down(col);
}

/* NOTE:
 * Back compatible APIs, take bottom-left point as the origin
 * LineBuffer shift up, while contents shift down
 * Assumes new values will be placed in top row(=ROWS-1)
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::shift_up(int col) {
#pragma HLS inline
    shift_pixels_up(col);
}

/* NOTE:
 * Back compatible APIs, take bottom-left point as the origin
 * LineBuffer insert
 * Inserts a new value in bottom row(=0)
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::insert_bottom(T value, int col) {
#pragma HLS inline
    insert_top_row(value, col);
}

/* NOTE:
 * Back compatible APIs, take bottom-left point as the origin
 * LineBuffer insert
 * Inserts a new value in top row(=ROWS-1)
 */
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::insert_top(T value, int col) {
#pragma HLS inline
    insert_bottom_row(value, col);
}

#ifndef __SYNTHESIS__
template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::restore_val() {
    HLS_SIZE_T i, j;
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            val_t[i][j] = val[i][j];
        }
    }
}

template <int ROWS, int COLS, typename T>
void LineBuffer<ROWS, COLS, T, 1>::linebuffer_print(int col) {
    HLS_SIZE_T i;
    for (i = 0; i < ROWS; i++) {
        std::cout << "\n";
        std::cout << std::setw(20) << val[i][col];
    }
    std::cout << "\n\n";
}
#endif

#endif //_RESIZE_MEM_HPP_
