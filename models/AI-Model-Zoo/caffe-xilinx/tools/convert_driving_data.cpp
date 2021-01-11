// This program converts detection bounding box labels to Dataum proto buffers
// and save them in LMDB.
// Usage:
//   convert_detection_label [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;
using google::protobuf::Message;

DEFINE_bool(test_run, false, "If set to true, only generate 100 images.");
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, true,
    "Randomly shuffle the order of images and their labels");
DEFINE_bool(use_rgb, false, "use RGB channels");
//SCNN
//DEFINE_int32(resize_width, 1640, "Width images are resized to");
//DEFINE_int32(resize_height, 600, "Height images are resized to");
//TSD setup
DEFINE_int32(resize_width, 640, "Width images are resized to");
DEFINE_int32(resize_height, 480, "Height images are resized to");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_driving_data [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_detection_label");
    return 1;
  }

  bool is_color = !FLAGS_gray;
  std::ifstream infile(argv[2]);
  std::vector<std::pair<string, vector<CarBoundingBox_Ming> > > lines;
  // LOG(ERROR) << "lines.size(): " << lines.size() << "\n"; //seokju
  LOG(ERROR) << "argv[1]: " << argv[1] << "\n"; //seokju
  LOG(ERROR) << "argv[2]: " << argv[2] << "\n"; //seokju
  string filename;
  int num_labels;
  while (infile >> filename >> num_labels) {
//LOG(ERROR) << "filename: " << filename << "\n"; //seokju
      //LOG(ERROR) << "=============" << "\n";
      vector<CarBoundingBox_Ming> boxes;
      if (num_labels != 0) {
        for (int i=0; i<num_labels; i++) {
            float tmp;
            CarBoundingBox_Ming box;
            ///-------------------For TSD + caltech_lane-------------------
            //infile >> tmp;
            /*
            if(filename[1] == 'T'){
                //LOG(INFO) << "--------------------" ;
                infile >> tmp; box.set_xmin(tmp*0.5);//for 1280x1024 pic
                infile >> tmp; box.set_ymin(tmp*0.5);
                infile >> tmp; box.set_xmax(tmp*0.5);
                infile >> tmp; box.set_ymax(tmp*0.5);
            */
            //--------------------For SCNN ----------------
            //}else if(filename[1] == 'd'){
                infile >> tmp; box.set_xmin(tmp);//*0.39);//for 1640x590 pic
                infile >> tmp; box.set_ymin(tmp);//*0.868);
                infile >> tmp; box.set_xmax(tmp);//*0.39);
                //LOG(ERROR) << "xmax ： " << tmp << "\n";
                infile >> tmp; box.set_ymax(tmp);//*0.868);
                //LOG(ERROR) << "ymax : " << tmp << "\n";
            /*
            }else{
                //LOG(ERROR) << filename;
                infile >> tmp; box.set_xmin(tmp);//for 640x480 pic
                infile >> tmp; box.set_ymin(tmp*1.07);
                infile >> tmp; box.set_xmax(tmp);
                infile >> tmp; box.set_ymax(tmp*1.07);
            }
            */
            ///------------------For TSD + caltech_lane--------------------
            /* For TSD
            infile >> tmp; box.set_xmin(tmp*0.5);//for 1280x1024 pic
            infile >> tmp; box.set_ymin(tmp*0.5);
            infile >> tmp; box.set_xmax(tmp*0.5);
            infile >> tmp; box.set_ymax(tmp*0.5);
            */
            int type;
            infile >> type; box.set_type(type);
            boxes.push_back(box);

            // LOG(ERROR) << "box.xmin(): " << box.xmin() << "\n"; //seokju
            // LOG(ERROR) << "box.ymin(): " << box.ymin() << "\n"; //seokju
            // LOG(ERROR) << "box.type(): " << box.type() << "\n"; //seokju
        }
        lines.push_back(std::make_pair(filename, boxes));
      }
  }

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  const char* db_path = argv[3];

  // bool generate_img = true;
  // std::string db_str(db_path);
  // if (db_str == "none") {
  //   generate_img = false;
  // }

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Open new db
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  // Open db
  LOG(INFO) << "Opening lmdb " << db_path;
  CHECK_EQ(mkdir(db_path, 0744), 0)
      << "mkdir " << db_path << "failed";
  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
      << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
      << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
      << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
      << "mdb_open failed. Does the lmdb already exist? ";

  // Storing to db
  string root_folder(argv[1]);
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  LOG(ERROR) << "Total to be processed: " << lines.size() << ".\n";
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    DriveData data;
    const string image_path = root_folder + lines[line_id].first;
    data.set_car_img_source(image_path);
    const vector<CarBoundingBox_Ming>& bbs = lines[line_id].second;
    for (int i = 0; i < bbs.size(); i ++) {
      CarBoundingBox_Ming *box = data.add_car_boxes();
      box->CopyFrom(bbs[i]);
    }
    if (!ReadImageToDatum(image_path, 0,
        resize_height, resize_width, is_color, data.mutable_car_image_datum())) {
        LOG(ERROR) << "*******************" ;
      continue;
    }

    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.c_str());
    string value;
    data.SerializeToString(&value);
    string keystr(key_cstr);

    // Put in db
    mdb_data.mv_size = value.size();
    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
    mdb_key.mv_size = keystr.size();
    mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
        << "mdb_put failed";

    if (++count % 1000 == 0) {
      // Commit txn
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
          << "mdb_txn_commit failed";
      CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
          << "mdb_txn_begin failed";
      LOG(ERROR) << "Processed " << count << " files.";
    }

    if (FLAGS_test_run && count == 100) {
      break;
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    mdb_close(mdb_env, mdb_dbi);
    mdb_env_close(mdb_env);
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
