
"""!
    @brief Waits for the end of DPU processing.
   
    @details modes: 
         1. Blocking wait for specific ID. 
         2. Non-blocking wait for specific ID. 
         3. Blocking wait for any ID. 
         4. Non-blocking wait for any ID
   
    @param jobid_time: tuple[uint32_t, int], [job id, time], jobid: neg for any id, others for specific job id.  time: not used here
   
    @return status 0 for exit successfully, others for customized warnings or errors
"""
def wait(jobid_time):
    pass
