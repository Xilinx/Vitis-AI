# Test Data

This folder contains both a sample dataset and code to prepare full benchmark test.

## Sample Data

* `access.log`: a hundred lines of Apache HTTP server log.
* `geo.dat`: a small geo-ip dataset for smoke test.
* `golden.json`: the expected output using input files above.

## Full Benchmark Data

An 1.2GB Apache log file can be downloaded [here](http://www.almhuette-raith.at/apache-log/access.log).

To obtain a complete geo dataset for deployment or evaluation, you can download "GeoLite2 City" dataset from
[here](https://dev.maxmind.com/geoip/geoip2/geolite2/), and convert it into the required format of the kernel
with following steps:

- Download and move the file `GeoLite2-City.mmdb` to this folder.
- Run `make` in this folder.
