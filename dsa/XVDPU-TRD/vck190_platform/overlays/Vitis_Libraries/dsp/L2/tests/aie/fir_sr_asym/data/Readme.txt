This file exists purely to force perforce to have the holding directory.
Without forcing this, test scripts for a library element can fail
for a new user who has just sync'd the library elemetn because the script
may attempt to write output files to directories which do not yet exist.
