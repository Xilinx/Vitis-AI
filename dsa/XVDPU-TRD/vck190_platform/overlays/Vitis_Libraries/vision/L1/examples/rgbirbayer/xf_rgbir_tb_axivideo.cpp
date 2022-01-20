#include <iostream>
#include <fstream>
#include <strstream>
#include <opencv2/opencv.hpp>
#include "xf_rgbir_config_axivideo.h"
#include "common/xf_axi.hpp"
#include "common/xf_sw_utils.hpp"

void subtract_ref(cv::Mat input1, cv::Mat input2, cv::Mat& output) {
    for (int row = 0; row < input1.rows; row++) {
        for (int col = 0; col < input1.cols; col++) {
            int val1 = input1.at<CVTYPE>(row, col);
            int val2 = input2.at<CVTYPE>(row, col);
            int val = val1 - val2;

            if (val < 0) {
                val = 0;
            }
            output.at<CVTYPE>(row, col) = (CVTYPE)val;
        }
    }
}

void wgtd_subtract_ref(float wgts[4], cv::Mat input1, cv::Mat input2, cv::Mat& output) {
    for (int row = 0; row < input1.rows; row++) {
        for (int col = 0; col < input1.cols; col++) {
            int val1 = input1.at<CVTYPE>(row, col);
            int val2 = input2.at<CVTYPE>(row, col);

            if (BPATTERN == XF_BAYER_GR) {
                if ((((row & 0x0001) == 0) && ((col & 0x0001) == 0)) ||
                    (((row & 0x0001) == 1) && ((col & 0x0001) == 1))) {        // G Pixel
                    val2 = val2 * wgts[0];                                     // G has medium level of reduced weight
                } else if ((((row & 0x0001) == 0) && ((col & 0x0001) == 1))) { // R Pixel
                    val2 = val2 * wgts[1];                                     // R has lowest level of reduced weight
                } else if (((((row - 1) % 4) == 0) && (((col) % 4) == 0)) ||
                           ((((row + 1) % 4) == 0) && (((col - 2) % 4) == 0))) { // B Pixel
                    val2 = val2 * wgts[2];                                       // B has low level of reduced weight
                } else if ((((((row - 1) % 4)) == 0) && (((col - 2) % 4) == 0)) ||
                           (((((row + 1) % 4)) == 0) && (((col) % 4) == 0))) { // Calculated B Pixel
                    val2 = val2 * wgts[3];                                     // B has highest level of reduced weight
                }
            }
            if (BPATTERN == XF_BAYER_BG) {
                if ((((row & 0x0001) == 0) && ((col & 0x0001) == 1)) ||
                    (((row & 0x0001) == 1) && ((col & 0x0001) == 0))) {        // G Pixel
                    val2 = val2 * wgts[0];                                     // G has medium level of reduced weight
                } else if ((((row & 0x0001) == 1) && ((col & 0x0001) == 1))) { // R Pixel
                    val2 = val2 * wgts[1];                                     // R has lowest level of reduced weight
                } else if (((((row) % 4) == 0) && (((col) % 4) == 0)) ||
                           ((((row - 2) % 4) == 0) && (((col - 2) % 4) == 0))) { // B Pixel
                    val2 = val2 * wgts[2];                                       // B has low level of reduced weight
                } else if ((((((row) % 4)) == 0) && (((col - 2) % 4) == 0)) ||
                           (((((row - 2) % 4)) == 0) && (((col) % 4) == 0))) { // Calculated B Pixel
                    val2 = val2 * wgts[3];                                     // B has highest level of reduced weight
                }
            }
            int val = val1 - val2;

            if (val < 0) {
                val = 0;
            }
            output.at<CVTYPE>(row, col) = (CVTYPE)val;
        }
    }
}

CVTYPE interp1_ref(CVTYPE val1, CVTYPE val2, CVTYPE val) {
    int ret = val1 + val * (val2 - val1);
    if (ret < 0) {
        ret = 0;
    }
    return (CVTYPE)ret;
}

CVTYPE bilinear_interp(CVTYPE val0, CVTYPE val1, CVTYPE val2, CVTYPE val3) {
    int ret = 0.25 * (val0 + val1 + val2 + val3);

    return (CVTYPE)ret;
}

void apply_bilinear_ref(CVTYPE patch[3][3], CVTYPE& out) {
    /*CVTYPE val1 = interp1_ref(patch[1][0], patch[1][2], 2);
    CVTYPE val2 = interp1_ref(patch[0][1], patch[2][1], 2);
    CVTYPE res = interp1_ref(val1, val2, 2);
     */
    CVTYPE res = bilinear_interp(patch[0][1], patch[1][0], patch[1][2], patch[2][1]);

    out = res;
}

void ir_bilinear_ref(cv::Mat half_ir, cv::Mat& full_ir) {
    CVTYPE block_half_ir[3][3];

    for (int row = 0; row < half_ir.rows; row++) {
        for (int col = 0; col < half_ir.cols; col++) {
            for (int k = -1, ki = 0; k < 2; k++, ki++) {
                for (int l = -1, li = 0; l < 2; l++, li++) {
                    if (row + k >= 0 && row + k < half_ir.rows && col + l >= 0 && col + l < half_ir.cols) {
                        block_half_ir[ki][li] = (CVTYPE)half_ir.at<CVTYPE>(row + k, col + l);
                    } else {
                        block_half_ir[ki][li] = 0;
                    }
                }
            }

            CVTYPE out_pix = block_half_ir[1][1];
            if ((BPATTERN == XF_BAYER_BG) || (BPATTERN == XF_BAYER_RG)) {
                if ((((row & 0x0001) == 0) && (col & 0x0001) == 1) ||
                    (((row & 0x0001) == 1) && ((col & 0x0001) == 0))) // BG, RG Mode - Even row, odd column and
                {                                                     //				odd row, even column
                    apply_bilinear_ref(block_half_ir, out_pix);
                }
                full_ir.at<CVTYPE>(row, col) = out_pix;
            } else if ((BPATTERN == XF_BAYER_GR)) {
                if ((((row & 0x0001) == 0) && (col & 0x0001) == 1)) // GR Mode - Even row, odd column
                {
                    apply_bilinear_ref(block_half_ir, out_pix);
                }
                full_ir.at<CVTYPE>(row, col) = out_pix;
            } else {
                if ((((row & 0x0001) == 0) && (col & 0x0001) == 0) ||
                    (((row & 0x0001) == 1) && ((col & 0x0001) == 1))) // GB Mode - Even row, even column and
                {                                                     //				odd row, odd column
                    apply_bilinear_ref(block_half_ir, out_pix);
                }
                full_ir.at<CVTYPE>(row, col) = out_pix;
            }
        }
    }
}

template <int FROWS, int FCOLS>
void apply_filter_ref(CVTYPE patch[FROWS][FCOLS], float weights[FROWS][FCOLS], CVTYPE& out) {
    float out_pix = 0;
    for (int fh = 0; fh < FROWS; fh++) {
        for (int fw = 0; fw < FCOLS; fw++) {
            out_pix += (patch[fh][fw] * weights[fh][fw]);
        }
    }
    if (out_pix < 0) {
        out_pix = 0;
    }

    out = (CVTYPE)out_pix;
}

void rgb_ir_ref(cv::Mat in, cv::Mat& output_image, cv::Mat& output_image_ir) {
    CVTYPE block_rgb[5][5];
    CVTYPE block_ir[3][3];
    bool toggle_weights = 0;

    float R_IR_C1_wgts[5][5] = {{-1.0 / 32.0f, -1.0 / 32.0f, 0, -1.0 / 32.0f, -1.0 / 32.0f},
                                {0, 1.0 / 2.0f, -1.0 / 16.0f, -1.0 / 4.0f, 0},
                                {0, -1.0 / 32.0f, -1.0 / 16.0f, -1.0 / 32.0f, 0},
                                {0, -1.0 / 4.0f, -1.0 / 16.0f, 1.0 / 2.0f, 0},
                                {-1.0 / 32.0f, -1.0 / 32.0f, 0, -1.0 / 32.0f, -1.0 / 32.0f}};
    float R_IR_C2_wgts[5][5] = {{-1.0 / 32.0f, -1.0 / 32.0f, 0, -1.0 / 32.0f, -1.0 / 32.0f},
                                {0, -1.0 / 4.0f, -1.0 / 16.0f, 1.0 / 2.0f, 0},
                                {0, -1.0 / 32.0f, -1.0 / 16.0f, -1.0 / 32.0f, 0},
                                {0, 1.0 / 2.0f, -1.0 / 16.0f, -1.0 / 4.0f, 0},
                                {-1.0 / 32.0f, -1.0 / 32.0f, 0, -1.0 / 32.0f, -1.0 / 32.0f}};
    float B_at_R_wgts[5][5] = {{1.0 / 8.0f, 0, -1.0 / 8.0f, 0, 1.0 / 8.0f},
                               {0, 0, 0, 0, 0},
                               {1.0 / 8.0f, 0, -1.0 / 2.0f, 0, 1.0 / 8.0f},
                               {0, 0, 0, 0, 0},
                               {1.0 / 8.0f, 0, -1.0 / 8.0f, 0, 1.0 / 8.0f}};
    float IR_at_R_wgts[3][3] = {{1.0 / 4.0f, 0, 1.0 / 4.0f}, {0, -1.0 / 16.0f, 0}, {1.0 / 4.0f, 0, 1.0 / 4.0f}};
    float IR_at_B_wgts[3][3] = {{1.0 / 4.0f, 0, 1.0 / 4.0f}, {0, -1.0 / 16.0f, 0}, {1.0 / 4.0f, 0, 1.0 / 4.0f}};

    // Extracting a 5x5 block from input image
    for (int row = 0; row < output_image.rows; row++) {
        for (int col = 0; col < output_image.cols; col++) {
            for (int k = -2, ki = 0; k < 3; k++, ki++) {
                for (int l = -2, li = 0; l < 3; l++, li++) {
                    if (row + k >= 0 && row + k < output_image.rows && col + l >= 0 && col + l < output_image.cols) {
                        block_rgb[ki][li] = (CVTYPE)in.at<CVTYPE>(row + k, col + l);
                    } else {
                        block_rgb[ki][li] = 0;
                    }
                }
            }
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    block_ir[i][j] = block_rgb[i + 1][j + 1];
                }
            }

            CVTYPE out_pix = block_rgb[2][2], out_pix_ir = block_ir[1][1];

            if (BPATTERN == XF_BAYER_BG) {
                if (((((row - 2) % 4) == 0) && (((col) % 4) == 0)) ||
                    ((((row) % 4) == 0) &&
                     (((col - 2) % 4) == 0))) // BG Mode - This is even row, R location. Compute B here with 5x5 filter
                {
                    apply_filter_ref<5, 5>(block_rgb, B_at_R_wgts, out_pix); // B at R
                } else if (((row & 0x0001) == 1) && ((col & 0x0001) == 1))   // BG Mode - This is odd row, odd column,
                // hence IR location. Compute R here with 5x5
                // filter
                {
                    if (toggle_weights) {
                        apply_filter_ref<5, 5>(block_rgb, R_IR_C2_wgts,
                                               out_pix); // R at IR - Constellation-2 (Red on the left)
                        toggle_weights = 0;
                    } else {
                        apply_filter_ref<5, 5>(block_rgb, R_IR_C1_wgts,
                                               out_pix); // R at IR - Constellation-1 (Blue on the left)
                        toggle_weights = 1;
                    }
                }
                // IR-calculations
                if ((((row % 4) == 0) && (((col) % 4) == 0)) ||
                    ((((row - 2) % 4) == 0) &&
                     (((col - 2) % 4) == 0))) // BG Mode - This is even row, R location. compute IR with 3x3 filter
                {
                    apply_filter_ref<3, 3>(block_ir, IR_at_B_wgts, out_pix_ir); // IR at R
                } else if (((((row - 2) % 4) == 0) && (((col) % 4) == 0)) ||
                           ((((row) % 4) == 0) &&
                            (((col - 2) % 4) ==
                             0))) { // BG Mode - Even row, even column, B location, apply 3x3 IR filter

                    apply_filter_ref<3, 3>(block_ir, IR_at_R_wgts, out_pix_ir); // IR at B location
                }
            } else if (BPATTERN == XF_BAYER_GR) { // For GR format

                if ((((row - 1) % 4 == 0) &&
                     ((col - 2) % 4) == 0) || // GR Mode - This is R location. Compute B here with 5x5 filter
                    (((row + 1) % 4 == 0) && (col % 4) == 0)) {
                    // Apply the filter on the NxM_src_blk
                    apply_filter_ref<5, 5>(block_rgb, B_at_R_wgts, out_pix); // R at B
                } else if (((row & 0x0001) == 0) && ((col & 0x0001) == 1)) // GR Mode - This is even row, odd col, hence
                                                                           // IR location. Compute R here with 5x5
                                                                           // filter
                {
                    if (toggle_weights) {
                        apply_filter_ref<5, 5>(block_rgb, R_IR_C1_wgts,
                                               out_pix); // R at IR - Constellation-1 (Red on the left top)
                        toggle_weights = 0;
                    } else {
                        apply_filter_ref<5, 5>(block_rgb, R_IR_C2_wgts,
                                               out_pix); // R at IR - Constellation-2 (Blue on the left top)
                        toggle_weights = 1;
                    }
                }
                // IR-calculations
                if (((((row - 1) % 4) == 0) &&
                     (((col - 2) % 4) == 0)) || // GR Mode - Odd row, R location. Apply 3x3 IR filter
                    ((((row + 1) % 4) == 0) && ((col % 4) == 0))) {             //- Next Odd row, R location
                    apply_filter_ref<3, 3>(block_ir, IR_at_R_wgts, out_pix_ir); // Calculating IR at R location
                } else if (((((row - 1) % 4) == 0) &&
                            ((col % 4) == 0)) || // GR Mode - Odd row, B location, apply 3x3 IR filter
                           ((((row + 1) % 4) == 0) && (((col - 2) % 4) == 0))) { //- Next Odd row, B location
                    apply_filter_ref<3, 3>(block_ir, IR_at_B_wgts, out_pix_ir);  // Calculating IR at B location
                }
            }
            /*			else if (BPATTERN == XF_BAYER_GR){ //For GR format

                    if(((row & 0x0001 ) == 0) && (((col+1) % 4) == 0 )) //GR Mode - This is even row, B location.
            Compute R here with 5x5 filter
                    {
                            // Apply the filter on the NxM_src_blk
                            apply_filter_ref<5,5>(block_rgb, B_at_R_wgts, out_pix);	//R at B
                    }
                    else if(((row & 0x0001 ) == 1) && ((col & 0x0001) == 0 )) //GR Mode - This is odd row, even col,
            hence IR location. Compute B here with 5x5 filter
                    {
                            if(toggle_weights){
                                    apply_filter_ref<5,5>(block_rgb, R_IR_C2_wgts, out_pix);	//R at IR -
            Constellation-2 (Red on the left)
                                    toggle_weights = 0;
                            }
                            else{
                                    apply_filter_ref<5,5>(block_rgb, R_IR_C1_wgts, out_pix);	//R at IR -
            Constellation-1 (Blue on the left)
                                    toggle_weights = 1;
                            }
                    }
                    //IR-calculations
                    if (((row & 0x0001) == 0) && (((col-1) % 4) == 0)){//GR Mode - Even row, R location. Apply 3x3 IR
            filter

                            apply_filter_ref<3,3>(block_ir, IR_at_R_wgts, out_pix_ir);	//Calculating IR at R location
                    }
                    else if (((row & 0x0001) == 0) && ((col & 0x0001) == 1)){//GR Mode - Even row, odd column, B
            location, apply 3x3 IR filter

                            apply_filter_ref<3,3>(block_ir, IR_at_B_wgts, out_pix_ir);	//Calculating IR at B location
                    }
            }*/
            output_image.at<CVTYPE>(row, col) = out_pix;
            output_image_ir.at<CVTYPE>(row, col) = out_pix_ir;
        }
    }

    cv::imwrite("WithIR_RGGB_ref.png", output_image);
    cv::imwrite("Half_ir_ref.png", output_image_ir);
}

void ref_rgb_ir(cv::Mat in, cv::Mat& rggb_output_image, cv::Mat& output_image_ir, int in_rows, int in_cols) {
    cv::Mat rggb_out_ref, half_ir_out_ref;

    rggb_out_ref.create(in_rows, in_cols, CV_INTYPE);
    half_ir_out_ref.create(in_rows, in_cols, CV_INTYPE);
    float wgts[4] = {0.125, 0.5, 0.25, 0.03125}; // GR

    rgb_ir_ref(in, rggb_out_ref, half_ir_out_ref);
    ir_bilinear_ref(half_ir_out_ref, output_image_ir);
    // subtract_ref(rggb_out_ref, output_image_ir, rggb_output_image);
    wgtd_subtract_ref(wgts, rggb_out_ref, output_image_ir, rggb_output_image);

#ifndef __SYNTHESIS
#ifdef __DEBUG

    FILE* fp1 = fopen("rggb_with_ir_ref.txt", "w");
    for (int i = 0; i < in_rows; ++i) {
        for (int j = 0; j < in_cols; ++j) {
            CVTYPE val = rggb_out_ref.at<CVTYPE>(i, j);
            fprintf(fp1, "%d ", val);
        }
        fprintf(fp1, "\n");
    }
    fclose(fp1);

    FILE* fp4 = fopen("half_ir_out_ref.txt", "w");
    for (int i = 0; i < in_rows; ++i) {
        for (int j = 0; j < in_cols; ++j) {
            CVTYPE val = half_ir_out_ref.at<CVTYPE>(i, j);
            fprintf(fp4, "%d ", val);
        }
        fprintf(fp4, "\n");
    }
    fclose(fp4);

    FILE* fp5 = fopen("sub_out_img.txt", "w");
    for (int i = 0; i < in_rows; ++i) {
        for (int j = 0; j < in_cols; ++j) {
            CVTYPE val = rggb_output_image.at<CVTYPE>(i, j);
            fprintf(fp5, "%d ", val);
        }
        fprintf(fp5, "\n");
    }
    fclose(fp5);
#endif
#endif
}

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage : <exe> <input image> <input height> <input width>" << std::endl;
        return 1;
    }

    cv::Mat rggb_out, ir_out;
    cv::Mat rggb_out_ref, half_ir_out_ref, ir_out_ref, sub_out_img;

    int in_rows = atoi(argv[2]);
    int in_cols = atoi(argv[3]);

    std::ifstream in_img(argv[1], std::ios::binary);

    if (in_img.fail()) {
        fprintf(stderr, "ERROR: Cannot open input image file %s\n ", argv[1]);
        return EXIT_FAILURE;
    }

    in_img.seekg(0, in_img.end);

    int total_bytes = in_img.tellg();

    std::cout << total_bytes << std::endl;

    CVTYPE* img_in_buffer = new CVTYPE[total_bytes];

    in_img.seekg(0, in_img.beg);

    in_img.read((char*)img_in_buffer, total_bytes);

    cv::Mat img_in(in_rows, in_cols, CV_INTYPE, img_in_buffer);

    if (img_in.data == NULL) {
        std::cout << "Unable to read the input image" << std::endl;
        return 1;
    }

    cv::imwrite("input.png", img_in);

    //############################
    // Reference code
    // ###########################

    rggb_out_ref.create(in_rows, in_cols, CV_INTYPE);
    ir_out_ref.create(in_rows, in_cols, CV_INTYPE);

    ref_rgb_ir(img_in, rggb_out_ref, ir_out_ref, in_rows, in_cols);

#ifndef __SYNTHESIS
#ifdef __DEBUG
    FILE* fp2 = fopen("input.txt", "w");
    for (int i = 0; i < in_rows; ++i) {
        for (int j = 0; j < in_cols; ++j) {
            CVTYPE val = img_in.at<CVTYPE>(i, j);
            fprintf(fp2, "%d ", val);
        }
        fprintf(fp2, "\n");
    }
    fclose(fp2);

    FILE* fp3 = fopen("rggb_out_ref.txt", "w");
    for (int i = 0; i < in_rows; ++i) {
        for (int j = 0; j < in_cols; ++j) {
            CVTYPE val = rggb_out_ref.at<CVTYPE>(i, j);
            fprintf(fp3, "%d ", val);
        }
        fprintf(fp3, "\n");
    }
    fclose(fp3);

    FILE* fp9 = fopen("ir_out_ref.txt", "w");
    for (int i = 0; i < in_rows; ++i) {
        for (int j = 0; j < in_cols; ++j) {
            CVTYPE val = ir_out_ref.at<CVTYPE>(i, j);
            fprintf(fp9, "%d ", val);
        }
        fprintf(fp9, "\n");
    }
    fclose(fp9);

#endif
#endif
    cv::imwrite("ref_ir_output.png", ir_out_ref);
    cv::imwrite("ref_rggb_output.png", rggb_out_ref);

    //############################
    // End Reference code
    //############################

    // 6 represents 0
    // 7 represents -1
    // All other numbers represent inverse of their value raised to 2 powers (ex: -5 represents -(1/32) )

    char R_IR_C1_wgts[25] = {-5, -5, 6, -5, -5, 6, 1, -4, -2, 6, 6, -5, -4, -5, 6, 6, -2, -4, 1, 6, -5, -5, 6, -5, -5};
    char R_IR_C2_wgts[25] = {-5, -5, 6, -5, -5, 6, -2, -4, 1, 6, 6, -5, -4, -5, 6, 6, 1, -4, -2, 6, -5, -5, 6, -5, -5};
    char B_at_R_wgts[25] = {3, 6, -3, 6, 3, 6, 6, 6, 6, 6, 3, 6, -1, 6, 3, 6, 6, 6, 6, 6, 3, 6, -3, 6, 3};
    char IR_at_R_wgts[9] = {2, 6, 2, 6, -4, 6, 2, 6, 2};
    char IR_at_B_wgts[9] = {2, 6, 2, 6, -4, 6, 2, 6, 2};
    char wgts[4] = {3, 1, 2, 5}; // Order for GR Format: G, R, B, Calculated B

    /*
            char R_IR_C1_wgts[25] = {-5,-5,0,-5,-5, 0,1,-4,-2,0, 0,-5,-4,-5,0, 0,-2,-4,1,0, -5,-5,0,-5,-5};
            char R_IR_C2_wgts[25] = {-5,-5,0,-5,-5, 0,-2,-4,1,0, 0,-5,-4,-5,0, 0,1,-4,-2,0, -5,-5,0,-5,-5};
            char B_at_R_wgts[25]  = {3,0,-3,0,3, 0,0,0,0,0, 3,0,-1,0,3, 0,0,0,0,0, 3,0,-3,0,3};
            char IR_at_R_wgts[9]  = {2,0,2, 0,-4,0, 2,0,2};
            char IR_at_B_wgts[9]  = {2,0,2, 0,-4,0, 2,0,2};
        char wgts[4]    	  = {3, 1, 2, 5}; // Order for GR Format: G, R, B, Calculated B
    */

    InStream Input;
    OutStream rggbOutput, irOutput;
    rggb_out.create(in_rows, in_cols, CV_INTYPE);
    ir_out.create(in_rows, in_cols, CV_INTYPE);

    xf::cv::cvMat2AXIvideoxf<NPC>(img_in, Input);

    rgbir_accel(Input, rggbOutput, irOutput, R_IR_C1_wgts, R_IR_C2_wgts, B_at_R_wgts, IR_at_R_wgts, IR_at_B_wgts, wgts,
                in_rows, in_cols);

    xf::cv::AXIvideo2cvMatxf<NPC>(rggbOutput, rggb_out);
    xf::cv::AXIvideo2cvMatxf<NPC>(irOutput, ir_out);

    cv::imwrite("rggb_out.png", rggb_out);
    cv::imwrite("ir_out.png", ir_out);

    // Result comparison
    cv::Mat diff_img_rggb(in_rows, in_cols, CV_INTYPE);
    cv::Mat diff_img_ir(in_rows, in_cols, CV_INTYPE);
    for (int i = 0; i < img_in.rows; i++) {
        for (int j = 0; j < img_in.cols; j++) {
            CVTYPE val_rggb = rggb_out.at<CVTYPE>(i, j);
            CVTYPE val_rggb_ref = rggb_out_ref.at<CVTYPE>(i, j);
            diff_img_rggb.at<CVTYPE>(i, j) = std::abs(val_rggb - val_rggb_ref);

            if (std::abs(val_rggb - val_rggb_ref) > 11) {
                std::cout << "\ni,j = " << i << ", " << j << std::endl;
                return -1;
            }

            CVTYPE val_ir = ir_out.at<CVTYPE>(i, j);
            CVTYPE val_ir_ref = ir_out_ref.at<CVTYPE>(i, j);
            diff_img_ir.at<CVTYPE>(i, j) = std::abs(val_ir - val_ir_ref);
        }
    }

    float err_per_rggb = 0, err_per_ir = 0;
    xf::cv::analyzeDiff(diff_img_rggb, ERROR_THRESHOLD_RGB, err_per_rggb);
    xf::cv::analyzeDiff(diff_img_ir, ERROR_THRESHOLD_IR, err_per_ir);
    cv::imwrite("diffImageRGGB.png", diff_img_rggb);
    cv::imwrite("diffImageIR.png", diff_img_ir);
    if (err_per_rggb > 1) {
        std::cout << "RGGB Test Failed" << std::endl;
        return 1;
    } else if (err_per_ir > 2) {
        std::cout << "IR Test Failed" << std::endl;
        return 1;
    } else {
        std::cout << "Test Passed" << std::endl;
        return 0;
    }
}
