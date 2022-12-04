#include <stdio.h>
#include "power_mon.hpp"

extern "C"{
	int main (int argc, char *argv[])
	{
		int volt = 0;
		int curr = 0;
		int power = 0;
		char volt_path[] = "/sys/class/hwmon/hwmon0/in1_input";
		char curr_path[] = "/sys/class/hwmon/hwmon0/curr1_input";
		char power_path[] =  "/sys/class/hwmon/hwmon0/power1_input";
		volt = get_device_volt(volt_path);
		curr = get_device_curr(curr_path);
		power = get_device_power(power_path);

		printf("device curr=%d, volt=%d, power=%d\n", curr,volt,power);
		return 0;
	}
}
