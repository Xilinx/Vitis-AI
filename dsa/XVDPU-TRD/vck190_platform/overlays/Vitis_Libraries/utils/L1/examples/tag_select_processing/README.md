# Tag-Select Processing

This example shows how to share work between multiple Process Units (PU) in tag-select fashion.

It asssumes that a few data will be updated by MPUs in the example.

The design of this example inludes three modules:

  1. Dispatch data to PUs by one stream to n distribution on tag select.
     Compared to distribution on round robin and load balance, here the width of output is same as the one of input.
```
          xf::common::utils_hw::streamOneToN<W_STRM,W_TAG>(
                         istrm,  e_istrm,
                         tg_strms[0], e_tg_strms[0],
                         data_inner_strms, e_data_inner_strms,
                         xf::common::utils_hw::TagSelectT());
```

  2. All PUs process data in parallel.
```
           process_mpu(
                        data_inner_strms, e_data_inner_strms,
                         new_data_strms,   e_new_data_strms);
```

  3. The new data in n streams from PUs are merged to a wide width stream.
```
          xf::common::utils_hw::streamNToOne<W_STRM, W_TAG>(
                        new_data_strms, e_new_data_strms,
                        tg_strms[1], e_tg_strms[1],
                        ostrm, e_ostrm,
                        xf::common::utils_hw::TagSelectT());
```

```
      stream_one_n                        process_mpu                      stream_n_to_one
  +-------------------+       +----------------------------------+       +-------------------+
  |                   |       |                                  |       |                   |
  |                   |       |                                  |       |                   |
  | one stream to n   |  -->  |  Multiple Process Units (MPU)    |  -->  | n streams to one  |
  |                   |       |                                  |       |                   |
  |                   |       |                                  |       |                   |
  +-------------------+       +----------------------------------+       +-------------------+
```

This ``update_mpu`` has the followint internal structure:

```
               split                    merge
               1-->8                    8-->1

          data_inner_strms[8]       new_data_strms[8]
                   |---------> PU0 --------->|
                   |                         |
                   |---------> PU1 --------->|
                   |                         |
                   |---------> PU2 --------->|
                   |                         |
                   |---------> PU3 --------->|
  istrm -------->  |                         |-----> ostrm
                   |---------> PU4 --------->|
                   |                         |
                   |---------> PU5 --------->|
                   |                         |
                   |---------> PU6 --------->|
                   |                         |
                   |---------> PU7 --------->|
```
