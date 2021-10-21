# Multi-PU Kernel

This example demonstrates writing a multi-PU kernel with the Hardware Utility
Library.

The kernel takes its input from an AXI port, and writes the result to another.
With in the kernel, eight dummy units shares the work load by
round-robin distribution and load-balancing collection.

The design of this example inludes three modules:

   1. All data are loaded from AXI port and converted to a wide width stream.
   ```
           xf::common::utils_hw::axiToStream<BURST_LENTH, W_AXI,ap_uint<W_STRM> >(
                      in_buf,
                      axi_strm,
                      e_axi_strm,
                      num,
                      offset );
   ```

   2. Multiple Process Uints update data.

      1) Dispatch data to PUs by one stream to n distribution on round robin.
         The wide width stream are splitted to n streams and each feeds to a PU.
          ```
           xf::common::utils_hw::streamOneToN<W_STRM, W_PU,NPU>(
                       istrm,  e_istrm,
                       data_inner_strms, e_data_inner_strms,
                       xf::common::utils_hw::RoundRobinT());
         ```
      2) All PUs work in parallel, but some work and others sleep sometimes.
         ```
           process_mpu(
                        data_inner_strms, e_data_inner_strms,
                         new_data_strms,   e_new_data_strms);

      3) The updated data in n streams from PUs are merged to a wide width stream.
         ```
           xf::common::utils_hw::streamNToOne<W_PU, W_STRM,NPU>(
                        new_data_strms, e_new_data_strms,
                        ostrm, e_ostrm,
                        xf::common::utils_hw::Round_RobinT());
         ```

   3. New data stream are ouput to axi port.
         ```
           xf::common::utils_hw::stream<BURST_LENTH, W_AXI,ap_uint<W_STRM> >(
                        in_buf,
                        axi_strm,
                        e_axi_strm );
         ```

 ```
    axi_to_stream                                 update_mpu                                    stream_to_axi
  +----------------+       +--------------------------------------------------------+       +-------------------+
  |                |       |                                                        |       |                   |
  |                |       |                                                        |       |                   |
  | axi to stream  |  -->  | stream to n streams   --> MPU  ---> n streams to one   |  -->  |  stream to axi    |
  |                |       |                                                        |       |                   |
  |                |       |                                                        |       |                   |
  +----------------+       +--------------------------------------------------------+       +-------------------+
 ```

The detials inside ``update_mpu``(8 PUs) like this:

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

All PUs work in parellel at an ideal case as belows:

```
   PU0   ------------------------------------
   PU1   ------------------------------------
   PU2   --  --  --  --  --  --  --  --  --
   PU3     --  --  --  --  --  --  --  --  --
   PU4   ----    ----    ----    ----    ----
   PU5       ----    ----    ----    ----
   PU6       ----    ----    ----    ----
   PU7       ----    ----    ----    ----
```

Here, the mark(-) stands for work and blank does sleep(does nothing).

