#include <google/protobuf/util/message_differencer.h>
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace google::protobuf;

vitis::ai::proto::DpuModelResult get_msg(const string file) {
  vitis::ai::proto::DpuModelResult msg;
  fstream input(file, ios::in | ios::binary);
  if (!input) {
    cerr << file << " : File not found." << endl;
    abort();
  } else if (!msg.ParseFromIstream(&input)) {
    cerr << "Failed to parse " << file << endl;
  }
  return msg;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << "usage: compare_proto <proto_file> <proto_file> " << endl;
    return 1;
  }

  auto msg1 = get_msg(argv[1]);
  // cout << "msg1 = " << msg1.DebugString() << endl;

  auto msg2 = get_msg(argv[2]);
  // cout << "msg2 = " << msg2.DebugString() << endl;

  auto equal = util::MessageDifferencer::Equals(msg1, msg2);
  if (!equal) {
    cout << argv[1] << " and " << argv[2] << " are diff." << endl;
  }

  return equal ? 0 : 1;
}
