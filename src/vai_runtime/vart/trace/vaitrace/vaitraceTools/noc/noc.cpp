#include <stdio.h>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include "noc.hpp"

NOC::NOC(int freq)
	:nsu_sample_addr {}
{
	mon_freq = freq * 1000 * 1000;
	noc_counter = 0;
}

NOC::~NOC()
{
	if(data != NULL)
		free(data);

}

int NOC::devmem_read(int addr){
	int fd;
	int read_value;
	void* map_ret, *map_virtual;
	//volatile uint32_t* map_virtual;
	unsigned page_size, mapped_size, offset_in_page;
	fd = open("/dev/mem", O_RDWR | O_SYNC);
	if(fd < 0) {
		printf("Err: can not open /dev/mem \n");
		return -1;
	}

	mapped_size = page_size = getpagesize();
	offset_in_page = (unsigned)addr & (page_size - 1);
	if (offset_in_page + 32  > page_size) {
		mapped_size *= 2;
	}
	map_ret = mmap(NULL, mapped_size,  PROT_READ | PROT_WRITE, MAP_SHARED, fd, addr & ~(page_size - 1));
	if(map_ret  == MAP_FAILED) {
		printf("Err: mmap get addr failed !\n");
		return -1;
	} else if ( map_ret == NULL ) {
		printf("Err: mmap get a NULL !\n");
		return -1;
	}

	map_virtual = (char*)map_ret + offset_in_page;
	read_value =  *(volatile uint32_t*)map_virtual;

	if (munmap(map_ret, mapped_size) == -1) {
		printf("Err: munmap failed !\n");
	}
	close(fd);

	return read_value;

}

int NOC::devmem_write(int addr,int value){
	int fd;
	int read_value;
	void* map_ret, *map_virtual;
	unsigned page_size, mapped_size, offset_in_page;

	fd = open("/dev/mem", O_RDWR | O_SYNC);
	if(fd < 0) {
		printf("Err: can not open /dev/mem \n");
		return -1;
	}

	mapped_size = page_size = getpagesize();
	offset_in_page = (unsigned)addr & (page_size - 1);
	if (offset_in_page + 32  > page_size) {
		mapped_size *= 2;
	}
	map_ret = mmap(NULL, mapped_size,  PROT_READ | PROT_WRITE, MAP_SHARED, fd, addr & ~(page_size - 1));

	if(map_ret  == MAP_FAILED) {
		printf("Err: mmap get addr failed !\n");
		return -1;
	} else if ( map_ret == NULL ) {
		printf("Err: mmap get a NULL !\n");
		return -1;
	}
	
	map_virtual = (char*)map_ret + offset_in_page;
	*(volatile uint32_t*)map_virtual = value;
	
	read_value = *(volatile uint32_t*)map_virtual;

	
	if (munmap(map_ret, mapped_size) == -1) {
		printf("Err: munmap failed !\n");
	}
	close(fd);

	return read_value;

}


//MON function

int NOC::mon_lock(int addr) {
	int state;

	addr = addr + 0xC;
	state = devmem_write(addr, 1);
	
	return state;
}

int NOC::mon_unlock(int addr) {
	int state;

	addr = addr + 0xC;
	state = devmem_write(addr, 0xF9E8D7C6);

	return state;
}

double NOC::mon_get_tbs(int addr) {
	int tbs_0;
	double tbs;

	tbs_0 = devmem_read(addr + 0x4D4) & (0x1f);
	// 1 / mon_freq * pow(2,tbs_0)
	tbs = pow(2,tbs_0) / mon_freq;

	return tbs;	
}

void NOC::mon_set_tbs(int addr, double period) {
	int tb_scale;
	double tb_value;
	double act_period;

	tb_value = log(mon_freq * period) / log(2);
	tb_scale = int(tb_value + 0.5);

	std::cout << "DDR_MC_NoC Freq:" << mon_freq << "Hz"<< std::endl;
	std::cout << "timebase scale reg:" << tb_scale << std::endl;
	
	devmem_write(addr + 0x4D4, tb_scale);

	act_period = mon_get_tbs(addr);
	mon_period = act_period;

	std::cout << "Target Sampling Period:" << period << std::endl;
	std::cout << "Act Period:" << act_period << std::endl;	
}
/*
double NOC::mon_get_act_period(void) {
	return mon_period;
}
*/
void NOC::mon_setup(int addr,double period) {
	
	if((0.001 <= period) && (period <= 2)) {
		mon_set_tbs(addr, period);
		
		for(int i = 0; i < 4; i++) {
			create_nsu_base_addr(addr, i);
		}
		
	} else {
		std::cout << "mon period error" << std::endl;
	}
}


//NSU FUNCTION

void NOC::nsu_enable(int addr) {
	devmem_write(addr, 1);
}

void NOC::nsu_disable(int addr) {
	devmem_write(addr,0);
}

void NOC::nsu_setup(int addr){

	devmem_write(addr + 0x4,0xf8);
}

int NOC::nsu_sample(int addr) {
	int mon_1;
	int burst_acc;
	int byte_count;

	mon_1 = devmem_read(addr + 0x18);
	/*
	if((mon_1 & (1 << 31)) != 0) {
		return -1;
	}
	*/
	burst_acc = mon_1 & ~(1 << 31);
	byte_count = burst_acc * 128 / 8;
	return byte_count;
	
}

void NOC::create_nsu_base_addr(int addr_base, int nsu_number) {

	if (addr_base == ddrmc_physics_addr1) {
		nsu_sample_addr[nsu_indexs] = nsu_number * 64 + addr_base + 0x4D8;	
	} else if (addr_base == ddrmc_physics_addr3) {
		nsu_sample_addr[nsu_indexs] = nsu_number * 64 + addr_base + 0x4D8;
	} else {
		printf("Err: nsu sample physics addr error\n");
	}
	
	nsu_indexs ++;
}


// COLLECT FUNCTION

void collecting_thread(NOC* noc, double sample_interval_clks) {
	
	uint32_t i;

	while(noc->collecting) {
		auto t = std::chrono::steady_clock::now().time_since_epoch();
		noc->data[noc->noc_counter].time = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1,1>>>(t).count();

		for(i=0; i < 8; i++) {
			noc->data[noc->noc_counter].data[i] = noc->nsu_sample(noc->nsu_sample_addr[i]);
		}

		if(noc->noc_counter * sizeof(struct noc_record) == DATA_BUFFER_SIZE) {
			noc->collecting = false;
		}
		usleep(1000 * 1000 * sample_interval_clks); // clks = s tranfer to ms
                noc->noc_counter ++;

	}
	std::cout << "NOC Stop Collecting" << std::endl; 
}

void NOC::start_collect (double sample_interval_clks, void *_data){

	if (_data != NULL)
		data = (struct noc_record *)_data;
	else
		data = (struct noc_record *)malloc(DATA_BUFFER_SIZE);
       
	memset(data, 0, DATA_BUFFER_SIZE);
	
	if (data == NULL) {
        	std::cout << "Unable to alloc memory for data" << std::endl;
		exit(1);
	}

	collecting = true;
	noc_thread = std::thread(collecting_thread,this,sample_interval_clks);


}

void NOC::stop_collect(void) {
	collecting = false;
	noc_thread.join();

}

int NOC::pop_data(struct noc_record *d){
	if(noc_counter == 0)
		return -1;
	d->time = data[noc_counter-1].time;
	for(int i = 0; i < 8; i++) {
		d->data[i] = data[noc_counter - 1].data[i];
	}

	noc_counter--;
	return 0;
}
