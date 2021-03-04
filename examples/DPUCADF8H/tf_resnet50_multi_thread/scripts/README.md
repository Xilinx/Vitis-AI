# Inconsistent accuracies observed in multi-threaded applications 

In current release, multi-threaded applications (threads >1) are resulting in inconsistent inference accuracies. Single threaded applications produce correct accuracy results. 

## Work Around:
In Multi-threaded applications, to preserve accuracy, users can use the **disabledru.py** script provided in this directory. Use your current xmodel and run **disabledru.py**, and utilize the output xmodel file, and re-run with your multi-threaded application. This will fix the inconsistent inference accuracy. With this workaround, you may see performance degradation. 

To revert back xmodel to earlier state, **enabledru.py** script run, this will deliver proper performance. 

## Fix:
A fix will come in the next release for this issue. 
