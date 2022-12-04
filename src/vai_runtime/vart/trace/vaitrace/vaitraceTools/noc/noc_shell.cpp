#include <stdio.h>
#include <stdlib.h>
#include "noc.hpp"

extern "C" {

	NOC noc(1000);

	int noc_start(double interval_sec)
	{	
		uint32_t i;
		
		noc.mon_unlock(ddrmc_physics_addr1);
		noc.mon_unlock(ddrmc_physics_addr3);

		noc.mon_setup(ddrmc_physics_addr1, interval_sec);
		noc.mon_setup(ddrmc_physics_addr3, interval_sec);


		for (i = 0; i < noc.nsu_indexs; i++) {
			noc.nsu_setup(noc.nsu_sample_addr[i]);
			noc.nsu_enable(noc.nsu_sample_addr[i]);
		}

		noc.start_collect(interval_sec);
		return EXIT_SUCCESS;
	}

	int noc_stop(void)
	{
		noc.stop_collect();
		return EXIT_SUCCESS;

	}

	double noc_act_period(void)
	{
		return noc.mon_period;
	}

	int noc_pop_data(struct noc_record *d)
	{
		return noc.pop_data(d);
	}
}
