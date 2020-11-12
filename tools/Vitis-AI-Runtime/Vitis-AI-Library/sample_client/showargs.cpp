#include <iostream>
using namespace std;
int main(int argc, char * argv[]) {
    for(int i ; i < argc; ++i) {
       cout << "argv[" << i << "]: " << argv[i] << endl;
    }
}
