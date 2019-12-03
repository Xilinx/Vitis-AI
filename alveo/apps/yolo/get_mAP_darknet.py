# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if x == "-":
      # skip file check and allow empty string
      return ""

    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

class Bbox:
    def __init__(self, class_id, prob, w,h,x,y):
        self.class_id = class_id
        self.prob = prob
        self.w = w
        self.h = h
        self.x = x
        self.y = y

class PR_info:
    def __init__(self, precision=0.0, recall=0.0, tp=0.0,fp=0.0,fn=0.0):
        self.precision = precision
        self.recall = recall
        self.tp = tp
        self.fp = fp
        self.fn = fn

class Detection_info:
    def __init__(self, class_id=-1, prob=1.0, x=0.0,y=0.0,w=0.0,h=0.0, truth_flag=0, unique_truth_index=-1):
        self.class_id = class_id
        self.prob = prob
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.truth_flag = truth_flag
        self.unique_truth_index = unique_truth_index

def compute_iou(gt_detection, detection):

    gt_x_start = gt_detection.x - gt_detection.w/2
    gt_y_start = gt_detection.y - gt_detection.h/2
    gt_x_end = gt_detection.x + gt_detection.w/2
    gt_y_end = gt_detection.y + gt_detection.h/2

    x_start = detection.x - detection.w/2
    y_start = detection.y - detection.h/2
    x_end = detection.x + detection.w/2
    y_end = detection.y + detection.h/2

    intersection_w = min(gt_x_end, x_end) - max(gt_x_start, x_start)
    intersection_h = min(gt_y_end, y_end) - max(gt_y_start, y_start)

    if ((intersection_w < 0) or (intersection_h < 0)):
        intersection_val = 0
    else:
        intersection_val = intersection_w * intersection_h

    uniou_val = (gt_detection.w * gt_detection.h) + (detection.w * detection.h) - intersection_val

    return (intersection_val/uniou_val)


def list_lines(file_path):
    # open txt file lines to a list
    with open(file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def calc_detector_mAP(labels_fpga_basepath, labels_gt_basepath, num_classes, class_names, thresh_calc_avg_iou, iou_thresh):

    name_list = os.listdir(labels_fpga_basepath)
    detect_box_files = sorted([os.path.join(labels_fpga_basepath,name) for name in name_list])
    gt_box_files = sorted([os.path.join(labels_gt_basepath,name) for name in name_list])

    with open("test.txt", "w") as the_file:
        for i in range(len(name_list)):
            if(os.path.exists(gt_box_files[i])):
                name_new = name_list[i].split(".")[0]
                the_file.write("data/obj/"+name_new+".jpg\n")

    print("Number of label files found=", len(name_list))
    gt_class_hist = np.zeros(num_classes)
    detection_list = []
    unique_truth_count = 0
    avg_iou = 0.0
    tp_for_thresh = 0.0
    fp_for_thresh = 0.0

    for file_num in range(len(name_list)):
        gt_exists = os.path.exists(gt_box_files[file_num])
        dt_exists = os.path.exists(detect_box_files[file_num])

        if(gt_exists == False or dt_exists== False):
            continue


        list_detections_gt = list_lines(gt_box_files[file_num])
        list_detections = list_lines(detect_box_files[file_num])
#        print("file:",name_list[file_num], " has ", len(list_detections_gt), " gt lines and ",  len(list_detections), " detections")
        gt_detections = []

        for line in list_detections_gt:
            tmp_class_name, x, y, w, h = line.split()
            detection = Detection_info(class_id = int(tmp_class_name),
                                           prob = 1,
                                           x = float(x),
                                           y = float(y),
                                           w= float(w),
                                           h= float(h))
            gt_class_hist[int(tmp_class_name)] = gt_class_hist[int(tmp_class_name)] + 1
            gt_detections.append(detection)

        num_gt_labels = len(gt_detections)

        for line in list_detections:
            tmp_class_name, confidence, x, y, w, h = line.split()
            if(float(confidence) > 0):
                detection = Detection_info(class_id = int(tmp_class_name),
                                           prob = float(confidence),
                                           x = float(x),
                                           y = float(y),
                                           w= float(w),
                                           h= float(h))

                truth_index = -1
                max_iou = 0.0

                for label_num in range(num_gt_labels):
                    current_iou = compute_iou(gt_detections[label_num], detection)

                    if((current_iou > iou_thresh) and (gt_detections[label_num].class_id == detection.class_id)):
                        if(current_iou > max_iou):
                            max_iou = current_iou
                            truth_index = unique_truth_count + label_num

                if (truth_index > -1):
                    detection.unique_truth_index = truth_index
                    detection.truth_flag = 1
                # changes may be needed related to difficult case

                detection_list.append(detection)
		#print("detection.prob=",detection.prob,"  thresh_calc_avg_iou:",  thresh_calc_avg_iou)
                if(detection.prob > thresh_calc_avg_iou):
                    if(truth_index > -1):
                        avg_iou += max_iou
                        tp_for_thresh += 1
                    else:
                        fp_for_thresh += 1
                #print("detection.prob=",detection.prob,"  thresh_calc_avg_iou:",  thresh_calc_avg_iou, " tp_for_thresh:", tp_for_thresh)
        unique_truth_count += num_gt_labels


    try:
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh)
    except ZeroDivisionError:
        avg_iou = 0

    detection_list.sort(key=lambda x: x.prob, reverse=True)
    utc_array = np.zeros(unique_truth_count)

    PR_info_list_of_lists =[]


    for rank in range(len(detection_list)):

        PR_info_list = []
        if(rank > 0):
            for class_um in range(num_classes):
                PR_info_elem_prev = PR_info_list_of_lists[rank - 1][class_um]
                PR_info_elem = PR_info(precision = PR_info_elem_prev.precision,
                                       recall = PR_info_elem_prev.recall,
                                       tp = PR_info_elem_prev.tp,
                                       fp = PR_info_elem_prev.fp,
                                       fn = PR_info_elem_prev.fn)
                PR_info_list.append(PR_info_elem)
        else:
            for class_um in range(num_classes):
                PR_info_elem = PR_info()
                PR_info_list.append(PR_info_elem)

        PR_info_list_of_lists.append(PR_info_list)

        detection = detection_list[rank]

        if(detection.truth_flag == 1):
            if(utc_array[detection.unique_truth_index] == 0):
                utc_array[detection.unique_truth_index] = 1
                PR_info_list_of_lists[rank][detection.class_id].tp += 1
        else:
            PR_info_list_of_lists[rank][detection.class_id].fp += 1

        for class_num in range(num_classes):
            tp = PR_info_list_of_lists[rank][class_num].tp
            fp = PR_info_list_of_lists[rank][class_num].fp
            fn = gt_class_hist[class_num] - tp
            PR_info_list_of_lists[rank][class_num].fn = fn

            if((tp+fp) > 0):
                PR_info_list_of_lists[rank][class_num].precision = tp/(tp+fp)
            else:
                PR_info_list_of_lists[rank][class_num].precision = 0

            if((tp+fn) > 0):
                PR_info_list_of_lists[rank][class_num].recall = tp/(tp+fn)
            else:
                PR_info_list_of_lists[rank][class_num].recall = 0


    mean_average_precision = 0.0

    for class_num in range(num_classes):

        avg_precision = 0
        for point in range(11):

            cur_recall = point * 0.1
            cur_precision = 0

            for rank in range(len(detection_list)):
                if((PR_info_list_of_lists[rank][class_num].recall >= cur_recall) and
                   (PR_info_list_of_lists[rank][class_num].precision > cur_precision)):

                       cur_precision = PR_info_list_of_lists[rank][class_num].precision

            avg_precision += cur_precision

        avg_precision = avg_precision/11
        print("class_id = ",class_num, " name = ", class_names[class_num], " ap = " , avg_precision*100)
        mean_average_precision += avg_precision

    print("tp_for_thresh= ",tp_for_thresh, " fp_for_thresh= ", fp_for_thresh)

    try:
        cur_precision = tp_for_thresh/(tp_for_thresh + fp_for_thresh)
    except ZeroDivisionError:
        cur_precision=0

    try:
        cur_recall = tp_for_thresh / (tp_for_thresh + (unique_truth_count - tp_for_thresh))
    except ZeroDivisionError:
        cur_recall = 0

    try:
        f1_score = 2.0 * cur_precision * cur_recall / (cur_precision + cur_recall)
    except ZeroDivisionError:
        f1_score = 0

    print("for thresh = " ,thresh_calc_avg_iou," precision = " , cur_precision, " recall = " ,cur_recall, " F1-score =  ",  f1_score)

    print("for thresh = " ,thresh_calc_avg_iou, " TP = ", tp_for_thresh, " FP = ",fp_for_thresh," FN = ",unique_truth_count - tp_for_thresh, " average IoU = ",  avg_iou * 100)

    mean_average_precision = mean_average_precision / num_classes

    #print("\n mean average precision (mAP) =  \n", mean_average_precision, mean_average_precision*100)

    print ("Mean Average Precision (mAP) = ", mean_average_precision, round(mean_average_precision*100, 2))

    return mean_average_precision


#labels = "/wrk/acceleration/users/arun/MLsuite/apps/yolo/coco.names"
#coco_gt_base_path = "/wrk/acceleration/shareData/COCO_Dataset/labels/val2014/"
#fpga_base_path="/wrk/acceleration/shareData/COCO_Dataset/gpu_val_result_224/"
#fpga_base_path="/wrk/acceleration/shareData/COCO_Dataset/fpga_val_result_224"
#fpga_base_path="/wrk/acceleration/shareData/COCO_Dataset/fpga_val_result_new/"
#fpga_base_path = "/wrk/acceleration/shareData/COCO_Dataset/gpu_val_result/"
#fpga_base_path = "/wrk/acceleration/users/arun/darknet/data/obj/out/"
#labels = "C:\\Users\\arunkuma\\Documents\\Work\\XDNN_customers\\Yolo_Aliun\\coco.names"
#coco_gt_base_path = "C:\\Users\\arunkuma\\Documents\\Work\\XDNN_customers\\Yolo_Aliun\\labels\\labels\\val2014"
#fpga_base_path = "C:\\Users\\arunkuma\\Documents\\Work\\XDNN_customers\\Yolo_Aliun\\compressed_filename\\fpga_val_result"


def main():
    parser = argparse.ArgumentParser(description='calc_detector_mAP')
    parser.add_argument('--class_names_file', help='File with name of the classes in rows', required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--ground_truth_labels', help='Direcotry path containing grund truth lable files in darknet style',
                        required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--detection_labels', help="direcotry path detected lable files in darknet style",
                        required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--IOU_threshold', type=float, default=0.5, help='IOU threshold generally set to 0.5')
    parser.add_argument('--prob_threshold', type=float, default=0.1, help='threshold for calculation of f1 score')

    args = parser.parse_args()
    try:
        args_dict = vars(args)
    except:
        args_dict = args

    with open(args_dict['class_names_file']) as f:
        names = f.readlines()

    class_names = [x.strip() for x in names]
    print("Class names are  :", class_names)

    mAP = calc_detector_mAP(args_dict['detection_labels'], args_dict['ground_truth_labels'], len(class_names), class_names, args_dict['prob_threshold'], args_dict['IOU_threshold'])


if __name__ == '__main__':
    main()
