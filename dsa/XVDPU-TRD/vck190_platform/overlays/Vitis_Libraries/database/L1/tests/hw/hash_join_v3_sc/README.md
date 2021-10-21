# Hash-Join V3 HLS Test

This HLS case tests the `hashJoinV3` APIs of the library, which performs the hash-table build and probel in a single call.

Comparing to `hashJoinMPU`, which is actually the 2nd version of our JOIN API, this version requires more AXI ports,
and thus works better on HBM devices.

