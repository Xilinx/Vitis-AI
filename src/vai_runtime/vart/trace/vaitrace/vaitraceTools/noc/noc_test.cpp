#include <stdio.h>
#include <iostream>
#include <inttypes.h>
#include "noc.hpp"

int main (int argc, char *argv[])
{


	NOC noc(1000);

	noc.mon_unlock(ddrmc_physics_addr1);
	noc.mon_unlock(ddrmc_physics_addr3);

	noc.mon_setup(ddrmc_physics_addr1, 0.01);
	noc.mon_setup(ddrmc_physics_addr3, 0.01);	

	for(int i = 0; i < noc.nsu_indexs; i++) {
		noc.nsu_setup(noc.nsu_sample_addr[i]);
		noc.nsu_enable(noc.nsu_sample_addr[i]);
	}	

	auto t1 = std::chrono::steady_clock::now().time_since_epoch();
    	double t2 =  std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1,1>>>(t1).count();
    	printf("%.7f\n", t2);

	noc.start_collect(0.01);
	std::this_thread::sleep_for(std::chrono::seconds(5));
	noc.stop_collect();

	printf("noc record counter = %d\n", noc.noc_counter);
	
	for(int j = 0; j < 8; j ++) {
		printf("the value of noc.data = %d\n", noc.data[0].data[j]);
	}

	return EXIT_SUCCESS;


}


