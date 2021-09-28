# Load-Balance PU Example

This example shows how to share work between multiple Process Units (PU) in load-balance fashion.

It asssumes that a few data will be updated by MPUs in the example.

The design of this example inludes three modules:

  1. Dispatch data to PUs by one stream to n distribution on load balance. 
        The wide width stream are splitted to n streams and each feeds to a PU.
      
        ``` 
           xf::common::utils_hw::streamOneToN<W_STRM, W_PU,NPU>(
                       istrm,  e_istrm,
                       data_inner_strms, e_data_inner_strms,
                       xf::common::utils_hw::LoadBalanceT());
        ``` 

  2. All PUs process data in parallel.
        ``` 
           process_mpu( 
                        data_inner_strms, e_data_inner_strms,
                         new_data_strms,   e_new_data_strms);
        ```                                

  3. The new data in n streams from PUs are merged to a wide width stream.
        ``` 
           xf::common::utils_hw::streamNToOne<W_PU, W_STRM,NPU>(
                        new_data_strms, e_new_data_strms,
                        ostrm, e_ostrm,
                        xf::common::utils_hw::LoadBalanceT());
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
   
 The detials inside update_mpu(8 PUs) like this:
 
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

