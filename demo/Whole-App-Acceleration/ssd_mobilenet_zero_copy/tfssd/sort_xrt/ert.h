/*
 *  Copyright (C) 2019, Xilinx Inc
 *
 *  This file is dual licensed.  It may be redistributed and/or modified
 *  under the terms of the Apache 2.0 License OR version 2 of the GNU
 *  General Public License.
 *
 *  Apache License Verbiage
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  GPL license Verbiage:
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as
 *  published by the Free Software Foundation; either version 2 of the
 *  License, or (at your option) any later version.  This program is
 *  distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 *  License for more details.  You should have received a copy of the
 *  GNU General Public License along with this program; if not, write
 *  to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
 *  Boston, MA 02111-1307 USA
 *
 */

/**
 * DOC: Xilinx SDAccel Embedded Runtime definition
 *
 * Header file *ert.h* defines data structures used by Emebdded Runtime (ERT) and
 * XRT xclExecBuf() API.
 */

#ifndef _ERT_H_
#define _ERT_H_

#if defined(__KERNEL__)
# include <linux/types.h>
#else
# include <stdint.h>
#endif

/**
 * struct ert_packet: ERT generic packet format
 *
 * @state:   [3-0] current state of a command
 * @custom:  [11-4] custom per specific commands
 * @count:   [22-12] number of words in payload (data)
 * @opcode:  [27-23] opcode identifying specific command
 * @type:    [31-28] type of command (currently 0)
 * @data:    count number of words representing packet payload
 */
struct ert_packet {
  union {
    struct {
      uint32_t state:4;   /* [3-0]   */
      uint32_t custom:8;  /* [11-4]  */
      uint32_t count:11;  /* [22-12] */
      uint32_t opcode:5;  /* [27-23] */
      uint32_t type:4;    /* [31-28] */
    };
    uint32_t header;
  };
  uint32_t data[1];   /* count number of words */
};

/**
 * struct ert_start_kernel_cmd: ERT start kernel command format
 *
 * @state:           [3-0]   current state of a command
 * @stat_enabled:    [4]     enabled driver to record timestamp for various
 *                           states cmd has gone through. The stat data
 *                           is appended after cmd data.
 * @extra_cu_masks:  [11-10] extra CU masks in addition to mandatory mask
 * @count:           [22-12] number of words following header for cmd data. Not
 *                           include stat data.
 * @opcode:          [27-23] 0, opcode for start_kernel
 * @type:            [31-27] 0, type of start_kernel
 *
 * @cu_mask:         first mandatory CU mask
 * @data:            count-1 number of words representing interpreted payload
 *
 * The packet payload is comprised of reserved id field, a mandatory CU mask,
 * and extra_cu_masks per header field, followed by a CU register map of size
 * (count - (1 + extra_cu_masks)) uint32_t words.
 */
struct ert_start_kernel_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t stat_enabled:1;   /* [4]     */
      uint32_t unused:5;         /* [9-5]   */
      uint32_t extra_cu_masks:2; /* [11-10] */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  /* payload */
  uint32_t cu_mask;          /* mandatory cu mask */
  uint32_t data[1];          /* count-1 number of words */
};

/**
 * struct ert_init_kernel_cmd: ERT initialize kernel command format
 * this command initializes CUs by writing CU registers. CUs are
 * represented by cu_mask and extra_cu_masks.
 *
 * @state:           [3-0] current state of a command
 * @extra_cu_masks:  [11-10] extra CU masks in addition to mandatory mask
 * @count:           [22-12] number of words following header
 * @opcode:          [27-23] 0, opcode for init_kernel
 * @type:            [31-27] 0, type of init_kernel
 *
 * @cu_run_timeout   the configured CU timeout value in Microseconds
 *                   setting to 0 means CU should not timeout
 * @cu_reset_timeout the configured CU reset timeout value in Microseconds
 *                   when CU timeout, CU will be reset. this indicates
 *                   CU reset should be completed within the timeout value.
 *                   if cu_run_timeout is set to 0, this field is undefined.
 *
 * @cu_mask:         first mandatory CU mask
 * @data:            count-9 number of words representing interpreted payload
 *
 * The packet payload is comprised of reserved id field, 8 reserved fields,
 * a mandatory CU mask, and extra_cu_masks per header field, followed by a
 * CU register map of size (count - (9 + extra_cu_masks)) uint32_t words.
 */
struct ert_init_kernel_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t unused:6;         /* [9-4]  */
      uint32_t extra_cu_masks:2; /* [11-10]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  uint32_t cu_run_timeout;   /* CU timeout value in Microseconds */
  uint32_t cu_reset_timeout; /* CU reset timeout value in Microseconds */
  uint32_t reserved[6];      /* reserved for future use */

  /* payload */
  uint32_t cu_mask;          /* mandatory cu mask */
  uint32_t data[1];          /* count-9 number of words */
};

#define KDMA_BLOCK_SIZE 64   /* Limited by KDMA CU */
struct ert_start_copybo_cmd {
  uint32_t state:4;          /* [3-0], must be ERT_CMD_STATE_NEW */
  uint32_t unused:6;         /* [9-4] */
  uint32_t extra_cu_masks:2; /* [11-10], = 3 */
  uint32_t count:11;         /* [22-12], = 15, exclude 'arg' */
  uint32_t opcode:5;         /* [27-23], = ERT_START_COPYBO */
  uint32_t type:4;           /* [31-27], = ERT_DEFAULT */
  uint32_t cu_mask[4];       /* mandatory cu masks */
  uint32_t reserved[4];      /* for scheduler use */
  uint32_t src_addr_lo;      /* low 32 bit of src addr */
  uint32_t src_addr_hi;      /* high 32 bit of src addr */
  uint32_t src_bo_hdl;       /* src bo handle, cleared by driver */
  uint32_t dst_addr_lo;      /* low 32 bit of dst addr */
  uint32_t dst_addr_hi;      /* high 32 bit of dst addr */
  uint32_t dst_bo_hdl;       /* dst bo handle, cleared by driver */
  uint32_t size;             /* size in bytes */
  void     *arg;             /* pointer to aux data for KDS */
};

/**
 * struct ert_configure_cmd: ERT configure command format
 *
 * @state:           [3-0] current state of a command
 * @count:           [22-12] number of words in payload (5 + num_cus)
 * @opcode:          [27-23] 1, opcode for configure
 * @type:            [31-27] 0, type of configure
 *
 * @slot_size:       command queue slot size
 * @num_cus:         number of compute units in program
 * @cu_shift:        shift value to convert CU idx to CU addr
 * @cu_base_addr:    base address to add to CU addr for actual physical address
 *
 * @ert:1            enable embedded HW scheduler
 * @polling:1        poll for command completion
 * @cu_dma:1         enable CUDMA custom module for HW scheduler
 * @cu_isr:1         enable CUISR custom module for HW scheduler
 * @cq_int:1         enable interrupt from host to HW scheduler
 * @cdma:1           enable CDMA kernel
 * @unused:25
 * @dsa52:1          reserved for internal use
 *
 * @data:            addresses of @num_cus CUs
 */
struct ert_configure_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t unused:8;         /* [11-4]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  /* payload */
  uint32_t slot_size;
  uint32_t num_cus;
  uint32_t cu_shift;
  uint32_t cu_base_addr;

  /* features */
  uint32_t ert:1;
  uint32_t polling:1;
  uint32_t cu_dma:1;
  uint32_t cu_isr:1;
  uint32_t cq_int:1;
  uint32_t cdma:1;
  uint32_t dataflow:1;
  uint32_t unusedf:24;
  uint32_t dsa52:1;

  /* cu address map size is num_cus */
  uint32_t data[1];
};

/**
 * struct ert_configure_sk_cmd: ERT configure soft kernel command format
 *
 * @state:           [3-0] current state of a command
 * @count:           [22-12] number of words in payload (13 DWords)
 * @opcode:          [27-23] 1, opcode for configure
 * @type:            [31-27] 0, type of configure
 *
 * @start_cuidx:     start index of compute units
 * @sk_type:         type of soft kernel.
 * @num_cus:         number of compute units in program
 * @sk_size:         size in bytes of soft kernel image
 * @sk_name:         symbol name of soft kernel
 * @sk_addr:         soft kernel image's physical address (little endian)
 */
struct ert_configure_sk_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t unused:8;         /* [11-4]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  /* payload */
  uint32_t start_cuidx:16;
  uint32_t sk_type:16;
  uint32_t num_cus;
  uint32_t sk_size;		/* soft kernel size */
  uint32_t sk_name[8];		/* soft kernel name */
  uint64_t sk_addr;
};

/**
 * struct ert_unconfigure_sk_cmd: ERT unconfigure soft kernel command format
 *
 * @state:           [3-0] current state of a command
 * @count:           [22-12] number of words in payload
 * @opcode:          [27-23] 1, opcode for configure
 * @type:            [31-27] 0, type of configure
 *
 * @start_cuidx:     start index of compute units
 * @num_cus:         number of compute units in program
 */
struct ert_unconfigure_sk_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t unused:8;         /* [11-4]  */
      uint32_t count:11;         /* [22-12] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };

  /* payload */
  uint32_t start_cuidx;
  uint32_t num_cus;
};

/**
 * struct ert_abort_cmd: ERT abort command format.
 *
 * @idx: The slot index of command to abort
 */
struct ert_abort_cmd {
  union {
    struct {
      uint32_t state:4;          /* [3-0]   */
      uint32_t unused:11;        /* [14-4]  */
      uint32_t idx:8;            /* [22-15] */
      uint32_t opcode:5;         /* [27-23] */
      uint32_t type:4;           /* [31-27] */
    };
    uint32_t header;
  };
};

/**
 * ERT command state
 *
 * @ERT_CMD_STATE_NEW:         Set by host before submitting a command to
 *                             scheduler
 * @ERT_CMD_STATE_QUEUED:      Internal scheduler state
 * @ERT_CMD_STATE_SUBMITTED:   Internal scheduler state
 * @ERT_CMD_STATE_RUNNING:     Internal scheduler state
 * @ERT_CMD_STATE_COMPLETE:    Set by scheduler when command completes
 * @ERT_CMD_STATE_ERROR:       Set by scheduler if command failed
 * @ERT_CMD_STATE_ABORT:       Set by scheduler if command abort
 * @ERT_CMD_STATE_TIMEOUT:     Set by scheduler if command timeout and reset
 * @ERT_CMD_STATE_NORESPONSE:  Set by scheduler if command timeout and fail to
 *                             reset
 */
enum ert_cmd_state {
  ERT_CMD_STATE_NEW = 1,
  ERT_CMD_STATE_QUEUED = 2,
  ERT_CMD_STATE_RUNNING = 3,
  ERT_CMD_STATE_COMPLETED = 4,
  ERT_CMD_STATE_ERROR = 5,
  ERT_CMD_STATE_ABORT = 6,
  ERT_CMD_STATE_SUBMITTED = 7,
  ERT_CMD_STATE_TIMEOUT = 8,
  ERT_CMD_STATE_NORESPONSE = 9,
  ERT_CMD_STATE_MAX, // Always the last one
};

struct cu_cmd_state_timestamps {
  uint64_t skc_timestamps[ERT_CMD_STATE_MAX]; // In nano-second
};

/**
 * Opcode types for commands
 *
 * @ERT_START_CU:       start a workgroup on a CU
 * @ERT_START_KERNEL:   currently aliased to ERT_START_CU
 * @ERT_CONFIGURE:      configure command scheduler
 * @ERT_EXEC_WRITE:     execute a specified CU after writing
 * @ERT_CU_STAT:        get stats about CU execution
 * @ERT_START_COPYBO:   start KDMA CU or P2P, may be converted to ERT_START_CU
 *                      before cmd reach to scheduler, short-term hack
 * @ERT_SK_CONFIG:      configure soft kernel
 * @ERT_SK_START:       start a soft kernel
 * @ERT_SK_UNCONFIG:    unconfigure a soft kernel
 */
enum ert_cmd_opcode {
  ERT_START_CU      = 0,
  ERT_START_KERNEL  = 0,
  ERT_CONFIGURE     = 2,
  ERT_EXIT          = 3,
  ERT_ABORT         = 4,
  ERT_EXEC_WRITE    = 5,
  ERT_CU_STAT       = 6,
  ERT_START_COPYBO  = 7,
  ERT_SK_CONFIG     = 8,
  ERT_SK_START      = 9,
  ERT_SK_UNCONFIG   = 10,
  ERT_INIT_CU       = 11,
};

/**
 * Command types
 *
 * @ERT_DEFAULT:        default command type
 * @ERT_KDS_LOCAL:      command processed by KDS locally
 * @ERT_CTRL:           control command uses reserved command queue slot
 * @ERT_CU:             compute unit command
 */
enum ert_cmd_type {
  ERT_DEFAULT = 0,
  ERT_KDS_LOCAL = 1,
  ERT_CTRL = 2,
  ERT_CU = 3,
};

/**
 * Soft kernel types
 *
 * @SOFTKERNEL_TYPE_EXEC:       executable
 * @SOFTKERNEL_TYPE_XCLBIN:     XCLBIN data file
 */
enum softkernel_type {
  SOFTKERNEL_TYPE_EXEC = 0,
  SOFTKERNEL_TYPE_XCLBIN = 1,
};

/**
 * Address constants per spec
 */
#define ERT_WORD_SIZE                      4          /* 4 bytes */
#define ERT_CQ_SIZE                        0x10000    /* 64K */
#ifndef ERT_BUILD_U50
# define ERT_CQ_BASE_ADDR                  0x190000
# define ERT_CSR_ADDR                      0x180000
#else
# define ERT_CQ_BASE_ADDR                  0x340000
# define ERT_CSR_ADDR                      0x360000
#endif

/**
 * The STATUS REGISTER is for communicating completed CQ slot indices
 * MicroBlaze write, host reads.  MB(W) / HOST(COR)
 */
#define ERT_STATUS_REGISTER_ADDR          (ERT_CSR_ADDR)
#define ERT_STATUS_REGISTER_ADDR0         (ERT_CSR_ADDR)
#define ERT_STATUS_REGISTER_ADDR1         (ERT_CSR_ADDR + 0x4)
#define ERT_STATUS_REGISTER_ADDR2         (ERT_CSR_ADDR + 0x8)
#define ERT_STATUS_REGISTER_ADDR3         (ERT_CSR_ADDR + 0xC)

/**
 * The CU DMA REGISTER is for communicating which CQ slot is to be started
 * on a specific CU.  MB selects a free CU on which the command can
 * run, then writes the 1<<CU back to the command slot CU mask and
 * writes the slot index to the CU DMA REGISTER.  HW is notified when
 * the register is written and now does the DMA transfer of CU regmap
 * map from command to CU, while MB continues its work. MB(W) / HW(R)
 */
#define ERT_CU_DMA_ENABLE_ADDR            (ERT_CSR_ADDR + 0x18)
#define ERT_CU_DMA_REGISTER_ADDR          (ERT_CSR_ADDR + 0x1C)
#define ERT_CU_DMA_REGISTER_ADDR0         (ERT_CSR_ADDR + 0x1C)
#define ERT_CU_DMA_REGISTER_ADDR1         (ERT_CSR_ADDR + 0x20)
#define ERT_CU_DMA_REGISTER_ADDR2         (ERT_CSR_ADDR + 0x24)
#define ERT_CU_DMA_REGISTER_ADDR3         (ERT_CSR_ADDR + 0x28)

/**
 * The SLOT SIZE is the size of slots in command queue, it is
 * configurable per xclbin. MB(W) / HW(R)
 */
#define ERT_CQ_SLOT_SIZE_ADDR             (ERT_CSR_ADDR + 0x2C)

/**
 * The CU_OFFSET is the size of a CU's address map in power of 2.  For
 * example a 64K regmap is 2^16 so 16 is written to the CU_OFFSET_ADDR.
 * MB(W) / HW(R)
 */
#define ERT_CU_OFFSET_ADDR                (ERT_CSR_ADDR + 0x30)

/**
 * The number of slots is command_queue_size / slot_size.
 * MB(W) / HW(R)
 */
#define ERT_CQ_NUMBER_OF_SLOTS_ADDR       (ERT_CSR_ADDR + 0x34)

/**
 * All CUs placed in same address space separated by CU_OFFSET. The
 * CU_BASE_ADDRESS is the address of the first CU. MB(W) / HW(R)
 */
#define ERT_CU_BASE_ADDRESS_ADDR          (ERT_CSR_ADDR + 0x38)

/**
 * The CQ_BASE_ADDRESS is the base address of the command queue.
 * MB(W) / HW(R)
 */
#define ERT_CQ_BASE_ADDRESS_ADDR          (ERT_CSR_ADDR + 0x3C)

/**
 * The CU_ISR_HANDLER_ENABLE (MB(W)/HW(R)) enables the HW handling of
 * CU interrupts.  When a CU interrupts (when done), hardware handles
 * the interrupt and writes the index of the CU that completed into
 * the CU_STATUS_REGISTER (HW(W)/MB(COR)) as a bitmask
 */
#define ERT_CU_ISR_HANDLER_ENABLE_ADDR    (ERT_CSR_ADDR + 0x40)
#define ERT_CU_STATUS_REGISTER_ADDR       (ERT_CSR_ADDR + 0x44)
#define ERT_CU_STATUS_REGISTER_ADDR0      (ERT_CSR_ADDR + 0x44)
#define ERT_CU_STATUS_REGISTER_ADDR1      (ERT_CSR_ADDR + 0x48)
#define ERT_CU_STATUS_REGISTER_ADDR2      (ERT_CSR_ADDR + 0x4C)
#define ERT_CU_STATUS_REGISTER_ADDR3      (ERT_CSR_ADDR + 0x50)

/**
 * The CQ_STATUS_ENABLE (MB(W)/HW(R)) enables interrupts from HOST to
 * MB to indicate the presense of a new command in some slot.  The
 * slot index is written to the CQ_STATUS_REGISTER (HOST(W)/MB(R))
 */
#define ERT_CQ_STATUS_ENABLE_ADDR         (ERT_CSR_ADDR + 0x54)
#define ERT_CQ_STATUS_REGISTER_ADDR       (ERT_CSR_ADDR + 0x58)
#define ERT_CQ_STATUS_REGISTER_ADDR0      (ERT_CSR_ADDR + 0x58)
#define ERT_CQ_STATUS_REGISTER_ADDR1      (ERT_CSR_ADDR + 0x5C)
#define ERT_CQ_STATUS_REGISTER_ADDR2      (ERT_CSR_ADDR + 0x60)
#define ERT_CQ_STATUS_REGISTER_ADDR3      (ERT_CSR_ADDR + 0x64)

/**
 * The NUMBER_OF_CU (MB(W)/HW(R) is the number of CUs per current
 * xclbin.  This is an optimization that allows HW to only check CU
 * completion on actual CUs.
 */
#define ERT_NUMBER_OF_CU_ADDR             (ERT_CSR_ADDR + 0x68)

/**
 * Enable global interrupts from MB to HOST on command completion.
 * When enabled writing to STATUS_REGISTER causes an interrupt in HOST.
 * MB(W)
 */
#define ERT_HOST_INTERRUPT_ENABLE_ADDR    (ERT_CSR_ADDR + 0x100)

/**
 * Interrupt controller base address
 * This value is per hardware BSP (XPAR_INTC_SINGLE_BASEADDR)
 */
#ifndef ERT_BUILD_U50
# define ERT_INTC_ADDR                     0x41200000
#else
# define ERT_INTC_ADDR                     0x00310000
#endif

/**
 * Look up table for CUISR for CU addresses
 */
#define ERT_CUISR_LUT_ADDR                (ERT_CSR_ADDR + 0x400)

/**
 * ERT exit command/ack
 */
#define	ERT_EXIT_CMD			  ((ERT_EXIT << 23) | ERT_CMD_STATE_NEW)
#define	ERT_EXIT_ACK			  (ERT_CMD_STATE_COMPLETED)

/**
 * State machine for both CUDMA and CUISR modules
 */
#define ERT_HLS_MODULE_IDLE               0x1
#define ERT_CUDMA_STATE                   (ERT_CSR_ADDR + 0x318)
#define ERT_CUISR_STATE                   (ERT_CSR_ADDR + 0x328)

/**
 * Interrupt address masks written by MB when interrupts from
 * CU are enabled
 */
#define ERT_INTC_IPR_ADDR                 (ERT_INTC_ADDR + 0x4)  /* pending */
#define ERT_INTC_IER_ADDR                 (ERT_INTC_ADDR + 0x8)  /* enable */
#define ERT_INTC_IAR_ADDR                 (ERT_INTC_ADDR + 0x0C) /* acknowledge */
#define ERT_INTC_MER_ADDR                 (ERT_INTC_ADDR + 0x1C) /* master enable */

/*
 * Helper functions to hide details of ert_start_copybo_cmd
 */
static inline void
ert_fill_copybo_cmd(struct ert_start_copybo_cmd *pkt, uint32_t src_bo,
  uint32_t dst_bo, uint64_t src_offset, uint64_t dst_offset, uint64_t size)
{
  pkt->state = ERT_CMD_STATE_NEW;
  pkt->extra_cu_masks = 3;
  pkt->count = 15;
  pkt->opcode = ERT_START_COPYBO;
  pkt->type = ERT_DEFAULT;
  pkt->cu_mask[0] = 0;
  pkt->cu_mask[1] = 0;
  pkt->cu_mask[2] = 0;
  pkt->cu_mask[3] = 0;
  pkt->src_addr_lo = src_offset;
  pkt->src_addr_hi = (src_offset >> 32) & 0xFFFFFFFF;
  pkt->src_bo_hdl = src_bo;
  pkt->dst_addr_lo = dst_offset;
  pkt->dst_addr_hi = (dst_offset >> 32) & 0xFFFFFFFF;
  pkt->dst_bo_hdl = dst_bo;
  pkt->size = size;
  pkt->arg = 0;
}
static inline uint64_t
ert_copybo_src_offset(struct ert_start_copybo_cmd *pkt)
{
  return (uint64_t)pkt->src_addr_hi << 32 | pkt->src_addr_lo;
}
static inline uint64_t
ert_copybo_dst_offset(struct ert_start_copybo_cmd *pkt)
{
  return (uint64_t)pkt->dst_addr_hi << 32 | pkt->dst_addr_lo;
}
static inline uint64_t
ert_copybo_size(struct ert_start_copybo_cmd *pkt)
{
  return pkt->size;
}

#define P2ROUNDUP(x, align)     (-(-(x) & -(align)))
static inline struct cu_cmd_state_timestamps *
ert_start_kernel_timestamps(struct ert_start_kernel_cmd *pkt)
{
  uint64_t offset = pkt->count * sizeof(uint32_t) + sizeof(pkt->header);
  /* Make sure the offset of timestamps are properly aligned. */
  return (struct cu_cmd_state_timestamps *)
    ((char *)pkt + P2ROUNDUP(offset, sizeof(uint64_t)));
}

#endif
