# Hash-Join V4 Build-Probe HLS Test

This HLS case tests the `hashBuildProbeV4` APIs of the library, which separates the hash-table build with one table
and probe with another into two steps, making it possible to probe a previously built hash-table with multiple different
tables or table shards.

