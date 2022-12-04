#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/utsname.h>

//char hwmon_path[] =  "/sys/class/hwmon"
int power = 0;
int curr = 0;
int volt = 0;

extern "C" {
	int get_device_power(char path[]){
		FILE* f_handle;
		f_handle = fopen(path,"rt");
		if (!f_handle) {
			printf("Err: open %s failure\n",path);
			exit(EXIT_FAILURE);
		}
		if (fscanf(f_handle, "%d", &power) == 0) {
			printf("Err: fscanf power failure\n");
		}
		fclose(f_handle);
		return power;
	}

	int get_device_curr(char* path) {
		FILE* f_handle;
                f_handle = fopen(path,"rt");
                if (!f_handle) {
                        printf("Err: open %s failure\n",path);
                        exit(EXIT_FAILURE);
                }
                if (fscanf(f_handle, "%d", &curr) == 0) {
			printf("Err: fscanf curr failure\n");
		}
		fclose(f_handle);
                return curr;
	}

	int get_device_volt(char* path){
		FILE* f_handle;
                f_handle = fopen(path,"rt");
                if(!f_handle){
                        printf("Err: open %s failure\n",path);
                        exit(EXIT_FAILURE);
                }
                if (fscanf(f_handle, "%d", &volt) == 0) {
			printf("Err: fscanf volt failure\n");
		}
		fclose(f_handle);
                return volt;
	}
}
