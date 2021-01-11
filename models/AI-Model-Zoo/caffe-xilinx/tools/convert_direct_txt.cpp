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
DEFINE_int32(width, 10, "Number of grids horizontally.");
DEFINE_int32(height, 10, "Number of grids vertically.");
DEFINE_int32(grid_dim, 8, "grid_dim x grid_dim number of pixels per each grid.");
DEFINE_int32(resize_width, 320, "Width images are resized to");
DEFINE_int32(resize_height, 320, "Height images are resized to");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_direct_txt [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_text");
    return 1;
  }


  // read bbs
  bool is_color = !FLAGS_gray;
  std::string root_folder(argv[1]);
  std::ifstream in(argv[2]);
  std::vector<std::pair<string, vector<int> > > lines;
  string filename;
  string jpgName, bbName;
  int tmp;
  int count = 0;
  while (in >> filename) {
    jpgName = root_folder + filename;
    // bbs
    in >> count;
    vector<int> bbs_landmark;
    for (int i = 0; i < count; i++) {
      for (int j = 0; j < 8; j ++) {
        in >> tmp;
        bbs_landmark.push_back(tmp);
      }      
    }
    lines.push_back(std::make_pair(jpgName, bbs_landmark));
    if (++count % 1000 == 0) {
      LOG(ERROR) << "Processed " << count << " bbs.";
    }
  }

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  const char* db_path = argv[3];
  std::string db_str(db_path);
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
  count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];

  LOG(ERROR) << "Total to be processed: " << lines.size() << ".\n";

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    // read image , set info
    DirectRegressionData data;
    data.set_text_label_resolution(FLAGS_grid_dim);
    data.set_text_shrink_factor(0.75);
    const string image_path = lines[line_id].first;
    data.set_text_img_source(image_path);
    if (!ReadImageToDatum(image_path, 0, 0, 0, is_color, "jpg", data.mutable_text_image_datum())) {
      continue;
    }

    const vector<int>& bbs_landmark = lines[line_id].second;
    for (int i = 0; i < bbs_landmark.size(); i += 8) {
      TextBox *text_box = data.add_text_boxes();
      text_box->set_x_1(bbs_landmark[i + 0]);
      text_box->set_y_1(bbs_landmark[i + 1]);
      text_box->set_x_2(bbs_landmark[i + 2]);
      text_box->set_y_2(bbs_landmark[i + 3]);
      text_box->set_x_3(bbs_landmark[i + 4]);
      text_box->set_y_3(bbs_landmark[i + 5]);
      text_box->set_x_4(bbs_landmark[i + 6]);
      text_box->set_y_4(bbs_landmark[i + 7]);
      text_box->set_img_width(resize_width);
      text_box->set_img_height(resize_height);
    }

    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id, lines[line_id].first.c_str());
    string value;
    data.SerializeToString(&value);
    string keystr(key_cstr);

    // Put in db
    mdb_data.mv_size = value.size();
    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
    mdb_key.mv_size = keystr.size();
    mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS) << "mdb_put failed";

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
