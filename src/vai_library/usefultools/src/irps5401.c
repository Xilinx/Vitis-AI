/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * pmbus_irps5401.c - adjust the fault limit of over current for running DPU
 */

#include <errno.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef __u8 u8;
typedef __u16 u16;
typedef __s8 s8;
typedef __s16 s16;

struct pmbus_dev {
  int fd;
  unsigned long funcs;
};

#define IOUT_OC_FAULT_LIMIT 0x46
/* Make sure you know the meaning of this value, otherwise DO NOT modify the
 * value*/
#define OC_FAULT_LIMIT_37A 0xF094

static const unsigned long i2c_func_pmbus_min = I2C_FUNC_SMBUS_BYTE_DATA |
                                                I2C_FUNC_SMBUS_WORD_DATA |
                                                I2C_FUNC_SMBUS_PROC_CALL;

static int pmbus_read_word_data(int fd, u16 cmd) {
  struct i2c_smbus_ioctl_data arg;
  u16 word;

  memset(&arg, 0, sizeof arg);
  arg.read_write = I2C_SMBUS_READ;
  arg.command = cmd;
  arg.size = I2C_SMBUS_WORD_DATA;
  arg.data = (union i2c_smbus_data *)&word;

  if (ioctl(fd, I2C_SMBUS, &arg) < 0) return -errno;

  return word;
}

static inline int pmbus_write_word_data(int fd, u16 cmd, u16 word) {
  struct i2c_smbus_ioctl_data arg;

  memset(&arg, 0, sizeof arg);
  arg.read_write = I2C_SMBUS_WRITE;
  arg.command = cmd;
  arg.size = I2C_SMBUS_WORD_DATA;
  arg.data = (union i2c_smbus_data *)&word;

  if (ioctl(fd, I2C_SMBUS, &arg) < 0) return -errno;
  return 0;
}

int main(int argc, char **argv) {
  int c;
  struct pmbus_dev dev;
  char *adapter = "/dev/i2c-4";
  int addr = 0x43;

  memset(&dev, 0, sizeof dev);
  dev.fd = open(adapter, O_RDWR);
  if (dev.fd < 0) {
    perror(adapter);
    fprintf(stderr, "Couldn't connect to I2C bus %s\n", adapter);
    return 1;
  }

  c = ioctl(dev.fd, I2C_FUNCS, &dev.funcs);
  if (c < 0) {
    perror(adapter);
    fprintf(stderr, "%s: Couldn't get funcs\n", adapter);
    return 1;
  }

  if ((dev.funcs & i2c_func_pmbus_min) != i2c_func_pmbus_min ||
      !(dev.funcs & (I2C_FUNC_SMBUS_READ_BLOCK_DATA | I2C_FUNC_I2C)) ||
      !(dev.funcs & (I2C_FUNC_SMBUS_BLOCK_PROC_CALL | I2C_FUNC_I2C))) {
    fprintf(stderr, "%s: Funcs don't support PMBus\n", adapter);
    return 1;
  }

  c = ioctl(dev.fd, I2C_SLAVE, addr);
  if (c < 0) {
    perror(adapter);
    fprintf(stderr, "Couldn't attach to device %#02x\n", addr);
    return 1;
  }

  /* Adjust the limit of output current */

  if (pmbus_read_word_data(dev.fd, IOUT_OC_FAULT_LIMIT) == OC_FAULT_LIMIT_37A) {
    return 0;
  } else {
    pmbus_write_word_data(dev.fd, IOUT_OC_FAULT_LIMIT, OC_FAULT_LIMIT_37A);
    if (pmbus_read_word_data(dev.fd, IOUT_OC_FAULT_LIMIT) !=
        OC_FAULT_LIMIT_37A) {
      fprintf(stderr, "OC fault limit is not 37A\n");
      return -1;
    }
  }

  return 0;
}
