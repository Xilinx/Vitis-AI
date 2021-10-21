# Hash-Join V4 HLS Test

This HLS case tests the `hashJoinV4` APIs of the library, which performs the hash-table build and probel in a single call.

Unlike `hashJoinV3` which uses on-chip memory for hash-table, this JOIN implementation uses on-chip memory as a bloom-filter.

