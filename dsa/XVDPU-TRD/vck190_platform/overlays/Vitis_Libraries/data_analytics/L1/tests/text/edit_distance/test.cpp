
#include <iostream>
#include "xf_data_analytics/text/editDistance.hpp"

#define N 35
#define M 35
#define BIT 6

void dut(hls::stream<ap_uint<BIT> >& len1_strm,
         hls::stream<ap_uint<64> >& query_strm,
         hls::stream<ap_uint<BIT> >& len2_strm,
         hls::stream<ap_uint<64> >& input_strm,
         hls::stream<ap_uint<BIT> >& max_ed_strm,
         hls::stream<bool>& i_e_strm,

         hls::stream<bool>& o_e_strm,
         hls::stream<bool>& o_match_strm) {
    xf::data_analytics::text::editDistance<N, M, BIT>(len1_strm, query_strm, len2_strm, input_strm, max_ed_strm,
                                                      i_e_strm, o_e_strm, o_match_strm);
}

int main() {
    std::string a = "Raymond Murin";
    std::string b = "Raymond Murfin";
    char str1[N], str2[M];
    for (int i = 0; i < N; i++) {
        if (i < a.length())
            str1[i] = a.at(i);
        else
            str1[i] = 0;
    }
    for (int i = 0; i < M; i++) {
        if (i < b.length())
            str2[i] = b.at(i);
        else
            str2[i] = 0;
    }

    ap_uint<N * 8> query_string;
    ap_uint<M * 8> input_string;
    for (int i = N - 1; i >= 0; i--) {
        query_string.range(8 * i + 7, i * 8) = str1[N - i - 1];
        std::cout << str1[N - i - 1];
    }
    std::cout << std::endl;
    for (int i = M - 1; i >= 0; i--) {
        input_string.range(8 * i + 7, i * 8) = str2[M - i - 1];
        std::cout << str2[M - i - 1];
    }
    std::cout << std::endl;

    hls::stream<ap_uint<BIT> > len1_strm;
    hls::stream<ap_uint<BIT> > len2_strm;
    hls::stream<ap_uint<64> > query_strm;
    hls::stream<ap_uint<64> > input_strm;
    hls::stream<ap_uint<BIT> > max_ed_strm;
    hls::stream<bool> i_e_strm;

    len1_strm.write(a.length());
    len2_strm.write(b.length());
    max_ed_strm.write(3);
    i_e_strm.write(false);
    i_e_strm.write(true);
    for (int j = 0; j < 1 + M / 8; j++) {
        ap_uint<64> t0 = ((j + 1) * 64 > 8 * M) ? query_string(8 * M - 1, 64 * j) : query_string(64 * j + 63, 64 * j);
        query_strm.write(t0);
    }
    for (int j = 0; j < 1 + M / 8; j++) {
        ap_uint<64> t0 = ((j + 1) * 64 > 8 * M) ? input_string(8 * M - 1, 64 * j) : input_string(64 * j + 63, 64 * j);
        input_strm.write(t0);
    }

    hls::stream<bool> o_e_strm;
    hls::stream<bool> o_match_strm;
    dut(len1_strm, query_strm, len2_strm, input_strm, max_ed_strm, i_e_strm, o_e_strm, o_match_strm);

    bool last = o_e_strm.read();
    while (!last) {
        last = o_e_strm.read();
        bool match = o_match_strm.read();
        if (match)
            std::cout << a << " --- " << b << " is matched. " << std::endl;
        else
            std::cout << a << " --- " << b << " is not matched. " << std::endl;
    }

    return 0;
}
