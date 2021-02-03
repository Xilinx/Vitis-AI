### Scripts for Inconsistencies in Accuracy observed for threads>1

An example scenario - Running the multi thread test with resnet50 xmodel taken from model Zoo provides good accuracy when ran with threads=1 but inconsistencies in accuracy are observed when ran with threads>1.

In such a scenario, if a user wants to gets good accuracy even while using threads>1, user can utilize disabledru.py script provided in this directory. User can provide current xmodel and run disabledru.py script and utilize the output xmodel from the script and re-run the test with threads>1, this should fix the inconsistencies observed in accuracy, however a dip in performance is noticed.

To revert back xmodel to earlier state, enabledru.py script can be used, where user would get best possible performance with threads>1 but with inconsistencies in accuracy.

