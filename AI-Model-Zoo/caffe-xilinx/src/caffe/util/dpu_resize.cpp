#include "caffe/util/dpu_resize.hpp"

dpu_resize::dpu_resize(uint8_t *ext_img_src, uint8_t *ext_img_dst,
                       struct _config ext_cfg) {
  img_src = ext_img_src;
  img_dst = ext_img_dst;
  cfg = ext_cfg;

  CQBIT = 15;
  IM_LINEAR = 0;
  IM_NEAREST = 1;
  IM_MAX = 2;
}

void dpu_resize::param_gen() {
  p_matrix = new struct _param *[cfg.dst_h];
  for (int y = 0; y < cfg.dst_h; y++) {
    p_matrix[y] = new struct _param[cfg.dst_w];
    for (int x = 0; x < cfg.dst_w; x++) {
      // Stage 0
      int32_t value_x_raw, value_y_raw;
      value_x_raw = x * cfg.scale_w;
      value_y_raw = y * cfg.scale_h;

      // Stage 1
      int32_t value_x, value_y;
      int32_t value_xp1, value_yp1;
      if (cfg.inter_mode == IM_LINEAR) {
        value_x = value_x_raw + (cfg.scale_w >> 1) - (1 << (CQBIT - 1));
        value_y = value_y_raw + (cfg.scale_h >> 1) - (1 << (CQBIT - 1));
      } else {
        value_x = value_x_raw;
        value_y = value_y_raw;
      }
      value_xp1 = value_x_raw + cfg.scale_w;
      value_yp1 = value_y_raw + cfg.scale_h;

      // Stage 2
      uint16_t start_x_tmp;
      uint16_t end_x_tmp;
      uint16_t start_y_tmp;
      uint16_t end_y_tmp;
      uint16_t frac_x_tmp[2];
      uint16_t frac_y_tmp[2];
      start_x_tmp =
          (value_x < 0) ? 0 : (value_x & (0xFFFFFFFF << CQBIT)) >> CQBIT;
      start_y_tmp =
          (value_y < 0) ? 0 : (value_y & (0xFFFFFFFF << CQBIT)) >> CQBIT;
      end_x_tmp = ceil(1.0 * value_xp1 / (1 << CQBIT));
      end_y_tmp = ceil(1.0 * value_yp1 / (1 << CQBIT));
      frac_x_tmp[0] = (value_x < 0) ? 0 : value_x & (~(0xFFFFFFFF << CQBIT));
      frac_x_tmp[1] = (value_x < 0) ? 0 : value_x & (~(0xFFFFFFFF << CQBIT));
      frac_y_tmp[0] = (value_y < 0) ? 0 : value_y & (~(0xFFFFFFFF << CQBIT));
      frac_y_tmp[1] = (value_y < 0) ? 0 : value_y & (~(0xFFFFFFFF << CQBIT));

      // Stage 3
      if (cfg.inter_mode == IM_LINEAR) {
        p_matrix[y][x].start_x =
            (start_x_tmp >= (cfg.src_w - 1)) ? (cfg.src_w - 1) : start_x_tmp;
        p_matrix[y][x].start_y =
            (start_y_tmp >= (cfg.src_h - 1)) ? (cfg.src_h - 1) : start_y_tmp;
        p_matrix[y][x].end_x = (start_x_tmp >= (cfg.src_w - 1))
                                   ? (cfg.src_w + 1)
                                   : start_x_tmp + 2;
        p_matrix[y][x].end_y = (start_y_tmp >= (cfg.src_h - 1))
                                   ? (cfg.src_h + 1)
                                   : start_y_tmp + 2;
      } else if (cfg.inter_mode == IM_NEAREST) {
        p_matrix[y][x].start_x =
            (start_x_tmp >= (cfg.src_w - 1)) ? (cfg.src_w - 1) : start_x_tmp;
        p_matrix[y][x].start_y =
            (start_y_tmp >= (cfg.src_h - 1)) ? (cfg.src_h - 1) : start_y_tmp;
        p_matrix[y][x].end_x = (start_x_tmp >= (cfg.src_w - 1))
                                   ? (cfg.src_w + 0)
                                   : start_x_tmp + 1;
        p_matrix[y][x].end_y = (start_y_tmp >= (cfg.src_h - 1))
                                   ? (cfg.src_h + 0)
                                   : start_y_tmp + 1;
      } else {
        /* Algorithm according to CAFFE
        p_matrix[y][x].start_x = (start_x_tmp >= cfg.src_w) ? cfg.src_w :
        start_x_tmp;
        p_matrix[y][x].start_y = (start_y_tmp >= cfg.src_h) ? cfg.src_h :
        start_y_tmp;
        */
        // Algorithm Opted by Qian
        p_matrix[y][x].start_x =
            (start_x_tmp >= (cfg.src_w - 1)) ? (cfg.src_w - 1) : start_x_tmp;
        p_matrix[y][x].start_y =
            (start_y_tmp >= (cfg.src_h - 1)) ? (cfg.src_h - 1) : start_y_tmp;
        p_matrix[y][x].end_x = (end_x_tmp >= cfg.src_w) ? cfg.src_w : end_x_tmp;
        p_matrix[y][x].end_y = (end_y_tmp >= cfg.src_h) ? cfg.src_h : end_y_tmp;
      }
      p_matrix[y][x].frac_x[0] = (start_x_tmp >= (cfg.src_w - 1))
                                     ? 1 << CQBIT
                                     : (1 << CQBIT) - frac_x_tmp[0];
      p_matrix[y][x].frac_x[1] =
          (start_x_tmp >= (cfg.src_w - 1)) ? 0 : frac_x_tmp[1];
      p_matrix[y][x].frac_y[0] = (start_y_tmp >= (cfg.src_h - 1))
                                     ? 1 << CQBIT
                                     : (1 << CQBIT) - frac_y_tmp[0];
      p_matrix[y][x].frac_y[1] =
          (start_y_tmp >= (cfg.src_h - 1)) ? 0 : frac_y_tmp[1];
    }
  }
}

void dpu_resize::calc() {
  param_gen();
  for (int dy = 0; dy < cfg.dst_h; ++dy) {
    for (int dx = 0; dx < cfg.dst_w; ++dx) {
      vector<int64_t> vrslt(cfg.src_c, 0);
      struct _param p = p_matrix[dy][dx];

      if ((p.start_x >= p.end_x) || (p.start_y >= p.end_y)) {
        for (int sc = 0; sc < cfg.src_c; ++sc) {
          vrslt[sc] = 0;
        }
      } else {
        for (int sy = p.start_y; sy < p.end_y; ++sy) {
          for (int sx = p.start_x; sx < p.end_x; ++sx) {
            for (int sc = 0; sc < cfg.src_c; ++sc) {
              if (cfg.inter_mode == IM_LINEAR) {
                vrslt[sc] = vrslt[sc] +
                            (int64_t(p.frac_x[sx - p.start_x]) *
                             p.frac_y[sy - p.start_y] *
                             img_src[sy * cfg.src_w * cfg.src_c +
                                     sx * cfg.src_c + sc]) /
                                (1 << (CQBIT + CQBIT - 4)) *
                                (1 << (CQBIT + CQBIT - 4));
              } else if (cfg.inter_mode == IM_NEAREST) {
                vrslt[sc] = int64_t(img_src[sy * cfg.src_w * cfg.src_c +
                                            sx * cfg.src_c + sc])
                            << (CQBIT + CQBIT);
              } else {
                vrslt[sc] = max(int64_t(img_src[sy * cfg.src_w * cfg.src_c +
                                                sx * cfg.src_c + sc])
                                    << (CQBIT + CQBIT),
                                vrslt[sc]);
              }
            }
          }
        }
      }
      for (int sc = 0; sc < cfg.dst_c; ++sc) {
        img_dst[dy * cfg.dst_w * cfg.dst_c + dx * cfg.dst_c + sc] =
            (vrslt[sc] + (1 << (CQBIT + CQBIT - 1))) >> (CQBIT + CQBIT);
      }
    }
  }
}

void dpu_resize::dump_config() {
  ofstream fout;
  fout.open("config.txt");
  fout << "scale_w\t\t= " << cfg.scale_w << endl;
  fout << "scale_h\t\t= " << cfg.scale_h << endl;
  fout << "src_w\t\t= " << cfg.src_w << endl;
  fout << "src_h\t\t= " << cfg.src_h << endl;
  fout << "src_c\t\t= " << cfg.src_c << endl;
  fout << "dst_w\t\t= " << cfg.dst_w << endl;
  fout << "dst_h\t\t= " << cfg.dst_h << endl;
  fout << "dst_c\t\t= " << cfg.dst_c << endl;
  fout << "inter_mode\t= " << cfg.inter_mode << endl;
  fout.close();
}

void dpu_resize::dump_img_src() {
  ofstream fout;
  fout.open("img_src.txt");
  for (int i = 0; i < cfg.src_h * cfg.src_w * cfg.src_c; i++)
    fout << (int)(img_src[i]) << endl;
  fout.close();
}

void dpu_resize::dump_img_dst() {
  ofstream fout;
  fout.open("img_dst.txt");
  for (int i = 0; i < cfg.dst_h * cfg.dst_w * cfg.dst_c; i++)
    fout << (int)(img_dst[i]) << endl;
  fout.close();
}

void dpu_resize::dump_param() {
  ofstream fout;
  fout.open("param.txt");

  for (int y = 0; y < cfg.dst_h; y++) {
    for (int x = 0; x < cfg.dst_w; x++) {
      fout << p_matrix[y][x].start_x << " " << p_matrix[y][x].end_x << " "
           << p_matrix[y][x].start_y << " " << p_matrix[y][x].end_y << " "
           << p_matrix[y][x].frac_x[0] << " " << p_matrix[y][x].frac_x[1] << " "
           << p_matrix[y][x].frac_y[0] << " " << p_matrix[y][x].frac_y[1] << " "
           << endl;
    }
  }
  fout.close();
}
