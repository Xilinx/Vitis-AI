### Benchmark Application

The test utilizes multiple threads and has the capability to utilize multiple PEs, to provide best possible performance for a given compiled xmodel.
As an example, resnet50 compiled xmodel is utilized.
Steps to run the application
1. make clean && make -j
2. Download the resnet50 xmodel from model Zoo and place it in the directory
3. Create a test image directory and pass the path in run script
4. ./run.sh

To observe best possible numbers, please consider running the test for 10K images or more.
We can vary number of threads, number of PEs in run.sh script by arguments -e and -c.

Note: Currently if threads is set to >1, inconsistencies in top-1, top-5 numbers are observed. This should be fixed in the upcoming release. Please look in scripts directory for more details on this inconsistency.
