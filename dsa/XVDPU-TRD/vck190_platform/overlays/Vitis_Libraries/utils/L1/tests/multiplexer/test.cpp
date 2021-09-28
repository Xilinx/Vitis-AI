#define SZ 100

#include <cstdint>
#include "xf_utils_hw/multiplexer.hpp"

struct Pix {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

void sender(char* in, hls::stream<ap_uint<16> >& s) {
    using namespace xf::common::utils_hw;

    uint64_t u;
    u = (uint64_t)(*in++) & 0xff;
    u |= ((uint64_t)(*in++) & 0xff) << 8;
    u |= ((uint64_t)(*in++) & 0xff) << 16;
    u |= ((uint64_t)(*in++) & 0xff) << 24;
    u |= ((uint64_t)(*in++) & 0xff) << 32;
    u |= ((uint64_t)(*in++) & 0xff) << 40;
    u |= ((uint64_t)(*in++) & 0xff) << 48;
    u |= ((uint64_t)(*in++) & 0xff) << 56;
    union {
        double d;
        uint64_t u;
    } un;
    un.u = u;
    double d = un.d;

    char c = *in++;

    Pix x;
    x.r = *in++;
    x.g = *in++;
    x.b = *in++;

    // Multiplexer<MUX_SENDER, 16> mux(s);
    auto mux = makeMux<MUX_SENDER>(s);

    // round 1
    mux.put(d);
    mux.put(c);
    mux.put(x);
    // round 2
    d /= 2;
    c += 1;
    x.r += 1;
    x.g += 1;
    x.b += 1;
    mux.put(d);
    mux.put(c);
    mux.put(x);
}

int emit(char* out, int offset, double d, char c, Pix x) {
#pragma HLS inline off
    int n = 0;
    union {
        double d;
        uint64_t u;
    } un;
    un.d = d;
    out[offset + (n++)] = (char)(un.u & 0xff);
    out[offset + (n++)] = (char)((un.u >> 8) & 0xff);
    out[offset + (n++)] = (char)((un.u >> 16) & 0xff);
    out[offset + (n++)] = (char)((un.u >> 24) & 0xff);
    out[offset + (n++)] = (char)((un.u >> 32) & 0xff);
    out[offset + (n++)] = (char)((un.u >> 40) & 0xff);
    out[offset + (n++)] = (char)((un.u >> 48) & 0xff);
    out[offset + (n++)] = (char)((un.u >> 56) & 0xff);

    out[offset + (n++)] = c;

    out[offset + (n++)] = x.r;
    out[offset + (n++)] = x.g;
    out[offset + (n++)] = x.b;

    return n;
}

void receiver(hls::stream<ap_uint<16> >& s, char* out) {
    using namespace xf::common::utils_hw;

    // Multiplexer<MUX_RECEIVER, 16> mux(s);
    auto mux = makeMux<MUX_RECEIVER>(s);

    // round 1
    double d = mux.get<double>();
    char c = mux.get<char>();
    Pix x = mux.get<Pix>();
    int o = emit(out, 0, d, c, x);

    // round 2, the simplier way to get
    mux.get(d);
    mux.get(c);
    mux.get(x);
    d *= 2;
    c -= 1;
    x.r -= 1;
    x.g -= 1;
    x.b -= 1;
    o += emit(out, o, d, c, x);

    // padding
    for (int i = 0; i < 4; ++i) {
#pragma HLS unroll
        out[o++] = 'a';
    }
    o += 4;
}

void dut(char in[SZ], char out[SZ]) {
#pragma HLS interface m_axi offset = slave bundle = gm0 port = in
#pragma HLS interface m_axi offset = slave bundle = gm1 port = out
#pragma HLS interface s_axilite bundle = control port = in
#pragma HLS interface s_axilite bundle = control port = out
#pragma HLS interface s_axilite bundle = control port = return
#pragma HLS dataflow
    hls::stream<ap_uint<16>, 32> s;
    sender(in, s);
    receiver(s, out);
}

#ifndef __SYNTHESIS__
#include <cstring>
#include <cstdio>
#include <memory>

int main(int argc, const char* argv[]) {
    auto din = std::make_unique<char[]>(SZ);
    auto dout = std::make_unique<char[]>(SZ);
    memset(din.get(), 0, SZ);
    memset(dout.get(), 0, SZ);

    double d = 3.1415926;
    char c = 'x';
    Pix x{125, 31, 83};

    size_t inc = sizeof(d);
    memcpy(din.get(), &d, inc);
    size_t total = inc;

    inc = 1;
    memcpy(din.get() + total, &c, 1);
    total++;

    inc = sizeof(x);
    memcpy(din.get() + total, &x, inc);
    total += inc;
    printf("din size %u\n", total);

    dut(din.get(), dout.get());

    int err = 0;
    if (memcmp(din.get(), dout.get(), total) == 0 && memcmp(din.get(), dout.get() + total, total) == 0) {
        printf("PASS!\n");
    } else {
        printf("FAIL!\n");
        err = 1;
        const char* op = dout.get();
        // round 1
        double od = *(reinterpret_cast<const double*>(op));
        printf("d: %lf, %lf\n", d, od);
        op += sizeof(double);

        char oc = *op;
        printf("c: %c, %c\n", c, oc);
        op++;

        Pix ox = *(reinterpret_cast<const Pix*>(op));
        printf("x: (%d, %d, %d), (%d, %d, %d)\n", (int)x.r, (int)x.g, (int)x.b, (int)ox.r, (int)ox.g, (int)ox.b);
        op += sizeof(Pix);

        // round 2
        od = *(reinterpret_cast<const double*>(op));
        printf("d: %lf, %lf\n", d, od);
        op += sizeof(double);

        oc = *op;
        printf("c: %c, %c\n", c, oc);
        op++;

        ox = *(reinterpret_cast<const Pix*>(op));
        printf("x: (%d, %d, %d), (%d, %d, %d)\n", (int)x.r, (int)x.g, (int)x.b, (int)ox.r, (int)ox.g, (int)ox.b);
        op += sizeof(Pix);

        printf("padding: %c %c %c %c\n", op[0], op[1], op[2], op[3]);
    }
    return err;
}
#endif
