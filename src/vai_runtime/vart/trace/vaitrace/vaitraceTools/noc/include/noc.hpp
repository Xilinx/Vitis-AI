#ifndef NOC_H
#define NOC_H

#include <cstdint>
#include <thread>
#include <array>

#define DATA_BUFFER_SIZE (4 * 1024 * 1024)
const int ddrmc_physics_addr1 = 0xf6210000;
const int ddrmc_physics_addr3 = 0xf64f0000;
struct noc_record {
        double time;
        unsigned int data[8];
};

class NOC 
{
public:

	int n_sample;
	int sleep_interval;
	int noc_counter = 0;
	double mon_period;
	uint64_t mon_freq;
	uint32_t nsu_indexs = 0;
	bool collecting = false;
	struct noc_record  *data = NULL;
	std::array<uint32_t,16> nsu_sample_addr;
	std::thread noc_thread;
	
	NOC(int freq);
	~NOC();
	int devmem_read(int addr);
        int devmem_write(int addr, int value);
	int mon_lock(int addr);
	int mon_unlock(int addr);
	void mon_set_tbs(int addr, double period);
	double mon_get_tbs(int addr);
	void mon_setup(int addr, double period);
	
	void nsu_enable(int addr);
	void nsu_disable(int addr);
	void nsu_setup(int addr);
	
        void create_nsu_base_addr(int ddr_number, int nsu_number);
        int get_sample_cout(void);
        int noc_data_process(int mons, int n_sample);
        void noc_loop_sleep_interval(int interval);
	int nsu_sample(int addr);
	int pop_data(struct noc_record *d);
        void start_collect(double sample_interval_clks, void *_data = NULL);
        void stop_collect(void);

};

#endif
