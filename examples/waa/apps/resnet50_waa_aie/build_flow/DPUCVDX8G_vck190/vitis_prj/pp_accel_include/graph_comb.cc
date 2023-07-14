#include "graph_conv.cc"
#include "graph.cpp"

//initialize and run the dataflow graph
#if defined(__AIESIM__) || defined(__X86SIM__)
int main(int argc, char** argv) {
    // Empty
    resize_norm.init();
    convgraph.init();
    resize_norm.update(resize_norm.a0, 104);
    resize_norm.update(resize_norm.a1, 107);
    resize_norm.update(resize_norm.a2, 123);
    resize_norm.update(resize_norm.a3, 0);
    resize_norm.update(resize_norm.b0, 8);
    resize_norm.update(resize_norm.b1, 8);
    resize_norm.update(resize_norm.b2, 8);
    resize_norm.update(resize_norm.b3, 0);
    convgraph.run();
    for (int outRowIdx = 0; outRowIdx < IMAGE_HEIGHT_OUT; outRowIdx++) {
        //resize_norm.update(resize_norm.row, outRowIdx);
        resize_norm.run(1);
        resize_norm.wait();
    }
    
    convgraph.end();
    resize_norm.end();

    return 0;
}
#endif

