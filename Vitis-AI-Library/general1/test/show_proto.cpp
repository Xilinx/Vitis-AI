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
  if (argc < 2) {
    cout << "usage: show_proto <proto_file>" << endl;
    return 1;
  }

  auto msg = get_msg(argv[1]);
  cout << argv[1] << " :" << endl;
  cout << msg.DebugString() << endl;

  return 0;
}
