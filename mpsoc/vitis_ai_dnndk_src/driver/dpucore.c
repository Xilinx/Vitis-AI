/*
 * Copyright (C) 2019 Xilinx, Inc.
 *
 * This software is licensed under the terms of the GNU General Public
 * License version 2, as published by the Free Software Foundation, and
 * may be copied, distributed, and modified under those terms.
 *
 */

#include "dpucore.h"

#define DEVICE_NAME "dpu"

typedef struct {
	uint8_t size;
	uint32_t *address;
}signature_handle_t;

typedef struct {
	uint8_t core_num;              // the numbers of DPU IPs
	uint16_t regs_size;            // register count in bytes for each DPU IP
	void *address[MAX_CORE_NUM];    // the base address of earch DPU IP
}ip_dpu_handle_t;

// Application functions
static unsigned long dpu_mem_alloc(uint32_t memsize);
static int dpu_mem_free(void *paddr);
static void dpu_regs_init(ioc_aol_init_t *data);
static int run_dpu(ioc_aol_run_t *prun);
static int run_softmax(ioc_aol_run_t *prun);
static void sync_mem_to_device(ioc_cache_ctrl_t *pmem);
static void sync_mem_from_device(ioc_cache_ctrl_t *pmem);
static void get_ips_device_handle(ioc_aol_device_handle_t *dev);
static int read_regs_32(void *address, uint32_t byte_len, uint32_t *output);

// Lowlevel functions
static int mask2id(uint32_t mask);
static uint64_t get_kernel_time(void);
static uint32_t field_mask_value(uint32_t val, uint32_t mask);

// Driver probe and remove
static int get_signature_base_addr(signature_handle_t *signature_handle);
static int get_dpu_cores_base_addr(uint32_t *signature_address, ip_dpu_handle_t *dpu_handle);
static int init_softmax(void *signature_address, struct device_node *pdpunode);
static int dpu_probe(struct platform_device *pdev);
static int dpu_remove(struct platform_device *pdev);
static void softmax_remove(void);

// Device interrupt
irqreturn_t dpu_isr(int irq, void *data);
irqreturn_t softmax_isr(int irq, void *data);

// Device operation
static long dpu_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
static int dpu_mmap(struct file *file, struct vm_area_struct *vma);
static ssize_t dpu_write(struct file *filp, const char __user *buf, size_t count, loff_t *f_pos);
static ssize_t dpu_read(struct file *filp, char __user *buf, size_t count, loff_t *ppos);
static int dpu_release(struct inode *inode, struct file *filp);
static int dpu_open(struct inode *inode, struct file *filp);

// Driver entry
static int __init dpu_init(void);
static void __exit dpu_exit(void);

static signature_handle_t g_signature_handle;
static ip_dpu_handle_t g_ip_dpu_handle;
static ioc_aol_device_handle_t g_aol_device_handle;

//DPU signature base address
unsigned long signature_addr = SIG_BASE;

uint accipmask = 0x0;
softmax_reg_t *gp_smfc_regs;

#if defined(CACHE_OFF)
int cache = 0; // whether use cache; 0:no, 1:yes
#else
int cache = 1; // whether use cache; 0:no, 1:yes
#endif

// the following parameters read from device tree
static int DPU_CORE_NUM;

atomic_t g_ref_count; //< dpu device open count
static struct device *dev_handler;
dpu_intrrupt_data_t *gp_dpu_ip_data[IP_MAX_COUNT];
dpu_intrrupt_data_t g_dpu_core_data[MAX_CORE_NUM];
dpu_intrrupt_data_t g_smfc_core_data[1];

struct miscdevice g_misc_device_register;

/*dpu registers*/
DPUReg *pdpureg;

struct list_head head_alloc; /*head of alloced memory block*/

struct semaphore memblk_lock;

/**
 * alloc a memory block from the available memory list.
 * @memsize : size of memory
 *
 *  RETURN: address of alloced memory;  NULL returned if no enough space exists
 */
static unsigned long dpu_mem_alloc(uint32_t memsize)
{
	void *virtaddr;
	dma_addr_t phy_addr;
	struct memblk_node *pnewnode;

	memsize = (memsize + (PAGE_SIZE - 1)) & ~(PAGE_SIZE - 1); //at least one page frame

	virtaddr = dma_alloc_coherent(dev_handler, memsize, &phy_addr, GFP_KERNEL);
	if (NULL != virtaddr) {
		pnewnode = kmalloc(sizeof(struct memblk_node), GFP_KERNEL);

		if (pnewnode) {
			pnewnode->virt_addr = (unsigned long)virtaddr;
			pnewnode->size = memsize;
			pnewnode->phy_addr = phy_addr;

			down(&memblk_lock);
			list_add(&pnewnode->list, &head_alloc);
			up(&memblk_lock);
		} else {
			dma_free_coherent(dev_handler, memsize, virtaddr, phy_addr);
			phy_addr = 0;
			dprint("kmalloc fail when adding memory node\n");
		}
		return phy_addr;
	} else {
		return 0;
	}
}

/**
 * Remove the memory block frome alloc list to the available
 *                memory list and merge with the neighbor node if necessary
 * @paddr :  address of memory block to be free
 */
static int dpu_mem_free(void *paddr)
{
	struct list_head *plist;
	struct memblk_node *p;

	down(&memblk_lock);

	list_for_each (plist, &head_alloc) {
		p = list_entry(plist, struct memblk_node, list);
		if (p->phy_addr == (dma_addr_t)paddr) {
			dma_free_coherent(dev_handler, p->size, (void *)p->virt_addr, p->phy_addr);
			list_del(&p->list);
			kfree(p);
			up(&memblk_lock);
			return 0;
		}
	}
	up(&memblk_lock);

	dprint("free memory failed,address=0x%p\n", paddr);

	return -ENXIO;
}

/**
 * dpu registers initialize
 * @channel: the dpu channel [0,DPU_CORE_NUM) need to be initialize,
 *			set all channel if the para is DPU_CORE_NUM
 */
static void dpu_regs_init(ioc_aol_init_t *data)
{
	int32_t i;
	uint32_t index = 0;
	uint32_t mask = data->core_mask;
	uint8_t *base_addr;
	uint8_t *p_regs;

	// IP name
	for (i = 0; i < IP_MAX_COUNT; i++) {
		if (data->ip_id == i) {
			break;
		}
		index += g_aol_device_handle.core_count[i];
	}
	if (i >= IP_MAX_COUNT) {
		return;
	}

	// core mask
	for (i = 0; i < 32; i++) {
		if (mask == 0) { // done
			break;
		} else if (mask & 0x01) { // Write regs
			base_addr = (uint8_t *)g_aol_device_handle.core_phy_addr[index];
			p_regs = ioremap((phys_addr_t)base_addr, MAX_REG_SIZE);
			for (i = 0; i < data->reg_count; i++) {
				iowrite32(data->regs[i].value, p_regs + data->regs[i].offset);
				if (data->regs_delay_us[i]) {
					udelay(data->regs_delay_us[i]);
				}
			}
			iounmap(p_regs);
		}

		mask >>= 1;
		index++;
	}
}

/**
 * Run dpu function
 * @prun : dpu run struct, contains the necessary address info
 *
 */
static int run_dpu(ioc_aol_run_t *prun)
{
	int i, ret = 0;
	uint8_t *pvalue;
	int dpu_core = mask2id(prun->core_mask);
	dpu_intrrupt_data_t *p_cur_core = &gp_dpu_ip_data[IP_ID_DPU][dpu_core];

	if (dpu_core >= DPU_CORE_NUM) {
		// should never get here
		dprint("ERR_CORE_NUMBER %d!\n", dpu_core);
		return -EINTR;
	}

	down(&p_cur_core->dpu_lock);
	prun->time_start = get_kernel_time();

	pvalue = ((uint8_t *)pdpureg)  + (dpu_core * 0x100);
	for (i = 0; i < prun->reg_count; i++) {
		iowrite32(prun->regs[i].value, pvalue + prun->regs[i].offset);
	}

	iowrite32(0x1, &(pdpureg->ctlreg[dpu_core].start));

	// wait for the dpu task to be finished
	ret = wait_event_interruptible_timeout(p_cur_core->waitqueue, p_cur_core->irq_flag == TRUE,
		prun->timeout * HZ);
	p_cur_core->irq_flag = FALSE;

	prun->time_end = get_kernel_time();
	up(&p_cur_core->dpu_lock);

	if (ret == 0) {
		dprint("[PID %d]Core %d Run timeout,failed to get finish interrupt!\n", current->pid, dpu_core);
	}

	return ret > 0 ? 0 : (ret == 0 ? -ETIMEDOUT : ret);
}

/**
 * softmax calculation acceleration using softmax IP
 * @para : softmax parameter structure
 *·
 * @return: 0 if successful; otherwise -errno
 */
static int run_softmax(ioc_aol_run_t *prun)
{
	int i;
	int ret = 0;
	dpu_intrrupt_data_t *p_cur_core = gp_dpu_ip_data[IP_ID_SOFTMAX];

	down(&p_cur_core->dpu_lock);

	// write softmax parameters
	for (i = 0; i < prun->reg_count; i++) {
		iowrite32(prun->regs[i].value, ((uint8_t *)gp_smfc_regs) + prun->regs[i].offset);
	}

	// start calculation
	iowrite32(1, &gp_smfc_regs->start);
	iowrite32(0, &gp_smfc_regs->start);

	ret = wait_event_interruptible_timeout(p_cur_core->waitqueue, p_cur_core->irq_flag == TRUE,
		prun->timeout * HZ);
	p_cur_core->irq_flag = FALSE;

	up(&p_cur_core->dpu_lock);

	if (ret == 0) {
		dprint("softmax timeout!\n");
	}

	return ret > 0 ? 0 : (ret == 0 ? -ETIMEDOUT : ret);
}

/**
 * flush memory range to ensure content is flushed to RAM
 * @pmem: memory fresh structure contains start virtual address and size
 */
static void sync_mem_to_device(ioc_cache_ctrl_t *pmem)
{
	dma_sync_single_for_device(dev_handler, pmem->addr_phy, pmem->size, DMA_BIDIRECTIONAL);
}

/**
 * invalid memory range to ensure following reading comes from RAM
 * @pmem: memory fresh structure contains start virtual address and size
 */
static void sync_mem_from_device(ioc_cache_ctrl_t *pmem)
{
	dma_sync_single_for_cpu(dev_handler, pmem->addr_phy, pmem->size, DMA_BIDIRECTIONAL);
}

static uint32_t field_mask_value(uint32_t val, uint32_t mask)
{
	int i;
	int max_bit = sizeof(uint32_t) * 8;
	int lowest_set_bit = max_bit - 1;

	/* Iterate through each bit of mask */
	for (i = 0; i < max_bit; i++) {
		/* If current bit is set */
		if ((mask >> i) & 1) {
			lowest_set_bit = i;
			break;
		}
	}

	return (val & mask) >> lowest_set_bit;
};

static void get_ips_device_handle(ioc_aol_device_handle_t *dev) {
	int i;
	int counter = 0;

	// Get signature base address
	dev->core_count[IP_ID_VER_REG] = 1;
	dev->core_phy_addr[counter++] = (uint64_t)g_signature_handle.address;

	// Get DPU cores base address
	dev->core_count[IP_ID_DPU] = g_ip_dpu_handle.core_num;
	for (i = 0; i < dev->core_count[IP_ID_DPU]; i++) {
		dev->core_phy_addr[counter++] = (uint64_t)g_ip_dpu_handle.address[i];
	}

	// Get Softmax base address
	dev->core_count[IP_ID_SOFTMAX] = 1;
	dev->core_phy_addr[counter++] = (uint64_t)DPU_EXT_SOFTMAX_BASE((unsigned long)g_signature_handle.address);
}

/**
 * 0,  Success. -1, Invalid DPU core number
 */
static int get_dpu_cores_base_addr(uint32_t *signature_address, ip_dpu_handle_t *dpu_handle) {
	int i;
	uint32_t signature_field = 0;

	read_regs_32(signature_address + 3, 4, &signature_field);
	dpu_handle->core_num = field_mask_value(signature_field, DPU_CORENUM_MASK);
	dpu_handle->regs_size = DPU_SIZE;

	if ((dpu_handle->core_num == 0) || (dpu_handle->core_num > MAX_CORE_NUM)) {
		dpr_init("Core number %d invalid!\n", DPU_CORE_NUM);
		return -1;
	}

	// Get each dpu base address
	for (i = 0; i < dpu_handle->core_num; i++) {
		dpu_handle->address[i] = (void *)DPU_BASE((unsigned long)signature_address);
	}

	return 0;
}

static int read_regs_32(void *address, uint32_t byte_len, uint32_t *output) {
	uint32_t i;
	uint32_t *p_regs;

	if (byte_len % 4 != 0) {
		dpr_init("read_regs_32 only support uint32_t read \n");
		return -1;
	}

	p_regs = ioremap((phys_addr_t)address, byte_len);
	for (i = 0; i < (byte_len >> 2); i++) {
		output[i] = ioread32(p_regs + i);
	}
	iounmap(p_regs);

	return 0;
}

static uint64_t get_kernel_time(void) {
	uint64_t time;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 10, 0)
	time = ktime_get();
#else
	time = ktime_get().tv64;
#endif
	return time;
}

static int mask2id(uint32_t mask) {
    int i;
    uint32_t test = 1;
    for (i = 0; i < 32; i++) {
        if ((mask & test) != 0) {
            break;
        }
        test <<= 1;
    }

    return i;
}

/**
 * dpu open function
 */
static int dpu_open(struct inode *inode, struct file *filp)
{
	if (atomic_read(&g_ref_count) == 0) {
	}

	atomic_inc(&g_ref_count);

	return 0;
}

/**
 * dpu close function
 * */
static int dpu_release(struct inode *inode, struct file *filp)
{
	struct list_head *plist, *nlist;
	struct memblk_node *p;

	if (atomic_dec_and_test(&g_ref_count)) {
		down(&memblk_lock);
		list_for_each_safe (plist, nlist, &head_alloc) {
			p = list_entry(plist, struct memblk_node, list);
			dma_free_coherent(dev_handler, p->size, (void *)p->virt_addr, p->phy_addr);
			list_del(&p->list);
			kfree(p);
		}
		INIT_LIST_HEAD(&head_alloc);
		up(&memblk_lock);
	}

	return 0;
}

static ssize_t dpu_read(struct file *filp, char __user *buf, size_t count, loff_t *ppos)
{
	return 0;
}

static ssize_t dpu_write(struct file *filp, const char __user *buf, size_t count, loff_t *f_pos)
{
	return 0;
}

/**
 * dpu mmap function
 */
static int dpu_mmap(struct file *file, struct vm_area_struct *vma)
{
	size_t size = vma->vm_end - vma->vm_start;

	if (!cache)
		vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

	if (remap_pfn_range(vma, vma->vm_start, vma->vm_pgoff, size, vma->vm_page_prot)) {
		return -EAGAIN;
	}
	return 0;
}

/**
 * dpu ioctl function
 */
static long dpu_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	int ret = 0;

	switch (cmd) {
		case DPU_IOCTL_MEM_ALLOC: { // memory alloc
			struct ioc_mem_alloc_t t;
			if (copy_from_user(&t, (void *)arg, sizeof(struct ioc_mem_alloc_t))) {
				return -EINVAL;
			}
			if (t.size == 0)
				return -EINVAL;
			t.addr_phy = dpu_mem_alloc(t.size);
			if (t.addr_phy == 0)
				return -ENOMEM;
			if (copy_to_user((void *)arg, &t, sizeof(struct ioc_mem_alloc_t)))
				return -EINVAL;
			break;
		}
		case DPU_IOCTL_MEM_FREE: { // memory free
			struct ioc_mem_free_t t;
			if (copy_from_user(&t, (void *)arg, sizeof(struct ioc_mem_free_t))) {
				return -EINVAL;
			}
			ret = dpu_mem_free((void *)t.addr_phy);

			break;
		}
		case DPU_IOCTL_RUN: { // run dpu
			ioc_aol_run_t t;
			if (copy_from_user(&t, (void *)arg, sizeof(ioc_aol_run_t))) {
				return -EINVAL;
			}

			if (t.ip_id == IP_ID_DPU) {
				ret = run_dpu(&t);
			} else if (t.ip_id == IP_ID_SOFTMAX) {
				ret = run_softmax(&t);
			} else {
				return -EINVAL;
			}

			if (copy_to_user((void *)arg, &t, sizeof(ioc_aol_run_t)))
				return -EINVAL;

			break;
		}
		case DPU_IOCTL_GET_DEV_HANDLE: {
			get_ips_device_handle(&g_aol_device_handle);
			if (copy_to_user((void *)arg, &g_aol_device_handle, sizeof(ioc_aol_device_handle_t))) {
				return -EINVAL;
			}

			break;
		}
		case DPU_IOCTL_INIT: { // reset dpu
			ioc_aol_init_t t;
			if (copy_from_user(&t, (void *)arg, sizeof(ioc_aol_init_t))) {
				return -EINVAL;
			}
			dpu_regs_init(&t);
			break;
		}
		case DPU_IOCTL_SYNC_TO_DEV: { // flush cache range by physical address
			ioc_cache_ctrl_t t;
			if (copy_from_user(&t, (void *)arg, sizeof(ioc_cache_ctrl_t)))
				return -EINVAL;
			sync_mem_to_device(&t);
			break;
		}
		case DPU_IOCTL_SYNC_FROM_DEV: { // invalidate cache range by physical address
			ioc_cache_ctrl_t t;
			if (copy_from_user(&t, (void *)arg, sizeof(ioc_cache_ctrl_t)))
				return -EINVAL;
			sync_mem_from_device(&t);
			break;
		}
		case DPU_IOCTL_READ_REGS: { // dpu capabilities
			ioc_aol_read_regs_t t;
			if (copy_from_user(&t, (void *)arg, 16)) {
				return -EINVAL;
			}
			if ((t.byte_size >> 2) >= READ_REG_DEFAULT_BUF_LEN) {
				dpr_init("Read regs size %d is overflow, shoule no more than %d\n", t.byte_size, READ_REG_DEFAULT_BUF_LEN);
				return -EINVAL;
			}
			if (read_regs_32((void *)t.phy_address, t.byte_size, t.out_buffer) != 0) {
				return -EINVAL;
			}
			if (copy_to_user((void *)arg, &t, sizeof(ioc_aol_read_regs_t))) {
				return -EINVAL;
			}
			break;
		}
		default: {
			ret = -EPERM;
			break;
		}
	}

	return ret;
}

/*dpu file operation define */
static struct file_operations dev_fops = {

	.owner = THIS_MODULE,
	.unlocked_ioctl = dpu_ioctl,
	.open = dpu_open,
	.release = dpu_release,
	.read = dpu_read,
	.write = dpu_write,
	.mmap = dpu_mmap,
};

/**
 * dpu interrupt service routine
 * when a task finished, dpu will generate a interrupt,
 * we can look up the IRQ No. to determine the channel
 */
irqreturn_t dpu_isr(int irq, void *data)
{
	int i = 0;
	dpu_intrrupt_data_t *p_cur_core = gp_dpu_ip_data[IP_ID_DPU];

	// Determine which channel generated the interrupt
	for (i = 0; i < DPU_CORE_NUM; i++) {
		if (irq == p_cur_core[i].irq_no) {
			// clear the interrupt
			iowrite32(0, &pdpureg->ctlreg[i].prof_en);
			iowrite32(0, &pdpureg->ctlreg[i].start);
			iowrite32((1 << i), &pdpureg->intreg.icr);
			udelay(1);
			iowrite32(0, &pdpureg->intreg.icr);

			// set the finish flag,record the time,and notify the waiting queue
			p_cur_core[i].irq_flag = TRUE;

			wake_up_interruptible(&p_cur_core[i].waitqueue);
		}
	}

	return IRQ_HANDLED;
}

/**
 * dpu extension modules isr
 * @irq  : interrupt number
 * @data : additional data
 */
irqreturn_t softmax_isr(int irq, void *data)
{
	dpu_intrrupt_data_t *p_cur_core = gp_dpu_ip_data[IP_ID_SOFTMAX];
	if (irq == p_cur_core->irq_no) {
		if (accipmask & DPU_EXT_SOFTMAX) {
			p_cur_core->irq_flag = TRUE;
			// clear smfc interrupt
			iowrite32(1, &gp_smfc_regs->clr);
			iowrite32(0, &gp_smfc_regs->clr);

			wake_up_interruptible(&p_cur_core->waitqueue);
		}
	}

	return IRQ_HANDLED;
}

static const char *dts_node_prefix[] = {
	"xilinx,",
	"xilinx, ",
	"Xilinx,",
	"Xilinx, ",
	"deephi,",
	"deephi, ",
	"Deephi,",
	"Deephi, ",
};

struct device_node *dpu_compatible_node(const char *compat)
{
	int idx=0, max=0;
	char dst_node[255];
	struct device_node *pdpu_node = NULL;

	if (strlen(compat)>128) {
		return NULL;
	}

	max = sizeof(dts_node_prefix)/sizeof(char *);
	for (idx=0; idx<max; idx++) {
		memset(dst_node, 0x0, sizeof(dst_node));
		sprintf(dst_node, "%s%s", dts_node_prefix[idx], compat);
		pdpu_node = of_find_compatible_node(NULL, NULL, dst_node);
		if (pdpu_node)
			break;
	}
	return pdpu_node;
};

/**
 * Initialize dpu extension modules
 * @pnode  : the dpu device tree node
 *
 * @return : 0 if OK; otherwise error number
 */
static int init_softmax(void *signature_address, struct device_node *pdpunode)
{
	int ret = 0;
	struct device_node *node;
	uint32_t softmax_valid;
	uint32_t signature_field;
	dpu_intrrupt_data_t *p_cur_core = gp_dpu_ip_data[IP_ID_SOFTMAX];

	// signature offset 9
	read_regs_32(((uint32_t *)signature_address) + 9, 4, &signature_field);
	softmax_valid = field_mask_value(signature_field, SOFTMAX_VLD_MASK);

	node = dpu_compatible_node("smfc");
	if (node && softmax_valid) {
		uint32_t reg_base, reg_size;
		dpr_init("Init softmax IP...\n");

		// register smfc interrupt isr
		p_cur_core->irq_no = irq_of_parse_and_map(node, 0);
		if (p_cur_core->irq_no < 0) {
			dpr_init("Softmax IRQ res not found!\n");
			return p_cur_core->irq_no;
		}
		ret = request_irq(p_cur_core->irq_no, (irq_handler_t)softmax_isr, 0, "dpu_smfc", NULL);
		if (ret != 0) {
			dpr_init("Request softmax IRQ %d failed!\n", p_cur_core->irq_no);
			return ret;
		} else {
			dpr_init("Request softmax IRQ %d successful.", p_cur_core->irq_no);
		}

		// map smfc register
		reg_base = DPU_EXT_SOFTMAX_BASE((unsigned long)signature_address);
		reg_size = DPU_EXT_SOFTMAX_SIZE;
		gp_smfc_regs = (ioremap(reg_base, reg_size));
		if (!gp_smfc_regs) {
			dpr_init("Map softmax registers error!\n");
			return -EINVAL;
		}

		// Init Softmax data
		init_waitqueue_head(&p_cur_core->waitqueue);
		sema_init(&p_cur_core->dpu_lock, 1);
		p_cur_core->irq_flag = FALSE;

		accipmask |= (softmax_valid ? DPU_EXT_SOFTMAX : 0);
		dpr_init("Init softmax IP done\n");
	}

	return ret;
}

/**
 * 0,  Success. -1, Invalid 'signature-addr' value
 */
static int get_signature_base_addr(signature_handle_t *signature_handle)
{
	const void *prop;
	struct device_node *pdpu_node;
	unsigned long signature_address = SIG_BASE;
	unsigned long base_addr_dtsi = 0;
	uint32_t signature_field = 0;

	pdpu_node = dpu_compatible_node("dpu");
	if (!pdpu_node) {
		dpr_init("Not found DPU device node!\n");
		return -1;
	}

	prop = of_get_property(pdpu_node, "base-addr", NULL);
	if (prop) {
		base_addr_dtsi = of_read_ulong(prop, 1);
	}
	if (base_addr_dtsi) {
		dpr_init("Found DPU signature addr = 0x%lx in device-tree\n", base_addr_dtsi);
		signature_address = base_addr_dtsi + 0x00F00000;
	}

	if (signature_address == SIG_BASE_NULL) {
		dpr_init("Signature address is NULL, please check.\n");
		return -1;
	}

	dpr_init("Checking DPU signature at addr = 0x%lx, \n", signature_address);
	read_regs_32((void *)signature_address, 4, &signature_field);
	if ((signature_field & SIG_MAGIC_MASK) == SIG_MAGIC) {
		signature_handle->size = field_mask_value(signature_field, SIG_SIZE_MASK);
		signature_handle->address = (void *)signature_address;
	} else {
		signature_handle->size = 0;
		signature_handle->address = 0;
		dpr_init("Invalid 'signature-addr' value specified in DPU device tree, please check.\n");
		return -1;
	}

	return 0;
}

/**
 * Platform probe method for the dpu driver
 * @pdev:	Pointer to the platform_device structure
 *
 * This function initializes the driver data structures and the hardware.
 *
 * @return:	0 on success and error value on failure
 */
static int dpu_probe(struct platform_device *pdev)
{
	int ret, i;
	struct device_node *pdpu_node, *dpucore_node;
	uint32_t signature_length = 0;
	uint32_t signature_field = 0;
	uint32_t *signature_va;
	dpu_intrrupt_data_t *p_cur_core;

	// init gp_dpu_ip_data
	gp_dpu_ip_data[IP_ID_DPU] = g_dpu_core_data;
	gp_dpu_ip_data[IP_ID_SOFTMAX] = g_smfc_core_data;
	gp_dpu_ip_data[IP_ID_FULLCONNECT] = g_smfc_core_data;

	p_cur_core = gp_dpu_ip_data[IP_ID_DPU];
	dev_handler = &(pdev->dev);
	dpucore_node = dpu_compatible_node("dpucore");

	pdpu_node = dpu_compatible_node("dpu");
	if (!pdpu_node) {
		dpr_init("Not found DPU device node!\n");
		return -ENXIO;
	}

	ret = get_signature_base_addr(&g_signature_handle);
	if (ret != 0) {
		return -ENXIO;
	}

	ret = get_dpu_cores_base_addr(g_signature_handle.address, &g_ip_dpu_handle);
	if (ret != 0) {
		return -EINVAL;
	}

	signature_length = g_signature_handle.size;
	signature_va = ioremap((phys_addr_t)g_signature_handle.address,
					signature_length * sizeof(signature_field));

	// offset 3
	read_regs_32(((uint32_t *)g_signature_handle.address) + 3, 4, &signature_field);
	DPU_CORE_NUM = field_mask_value(signature_field, DPU_CORENUM_MASK);

	// map the dpu Register, all DPUs share the same registers area
	pdpureg = (DPUReg *)ioremap((phys_addr_t)g_ip_dpu_handle.address[0], g_ip_dpu_handle.regs_size);
	if (!pdpureg) {
		dpr_init("Map DPU registers error!\n");
		return -ENXIO;
	}

	// init DPU data
	for (i = 0; i < DPU_CORE_NUM; i++) {
		init_waitqueue_head(&p_cur_core[i].waitqueue);
		sema_init(&p_cur_core[i].dpu_lock, 1);
		p_cur_core[i].irq_flag = FALSE;
	}

	// memory structure init
	sema_init(&memblk_lock, 1);
	INIT_LIST_HEAD(&head_alloc);

	// register interrupt service routine for DPU
	for (i = 0; i < DPU_CORE_NUM; i++) {
		p_cur_core[i].irq_no = dpucore_node? irq_of_parse_and_map(dpucore_node, i): platform_get_irq(pdev, i);

		if (p_cur_core[i].irq_no < 0) {
			dprint("IRQ resource not found for DPU core %d\n", i);
			return p_cur_core[i].irq_no;
		}

		ret = request_irq(p_cur_core[i].irq_no, (irq_handler_t)dpu_isr, 0, "dpu_isr", NULL);
		if (ret != 0) {
			dpr_init("Request IRQ %d failed!\n", p_cur_core[i].irq_no);
			return ret;
		} else {
		}
	}

	// Reset DPU
	iowrite32(0, &pdpureg->pmu.reset);
	udelay(1); // wait 1us
	iowrite32(0xFFFFFFFF, &pdpureg->pmu.reset);
	iowrite32(0xFF, &pdpureg->intreg.icr);
	udelay(1); // wait 1us
	iowrite32(0, &pdpureg->intreg.icr);

	// initialize extent modules
	init_softmax(g_signature_handle.address, pdpu_node);

	// Register the dpu device
	g_misc_device_register.name = DEVICE_NAME;
	g_misc_device_register.minor = MISC_DYNAMIC_MINOR;
	g_misc_device_register.fops = &dev_fops;
	g_misc_device_register.mode = S_IWUGO | S_IRUGO;
	return misc_register(&g_misc_device_register);
}

static void softmax_remove(void)
{
	dpu_intrrupt_data_t *p_cur_core = gp_dpu_ip_data[IP_ID_SOFTMAX];
	if (accipmask & DPU_EXT_SOFTMAX) {
		// clean smfc moudle
		iounmap(gp_smfc_regs);
		free_irq(p_cur_core->irq_no, NULL);
	}
}

/**
 * Platform remove method for the dpu driver
 * @pdev:	Pointer to the platform_device structure
 *
 * This function is called if a device is physically removed from the system or
 * if the driver module is being unloaded. It frees all resources allocated to
 * the device.
 *
 * @return:	0 on success and error value on failure
 */
static int dpu_remove(struct platform_device *pdev)
{
	int i;

	// remove dpu
	dpu_intrrupt_data_t *p_cur_core = gp_dpu_ip_data[IP_ID_DPU];
	misc_deregister(&g_misc_device_register);

	for (i = 0; i < DPU_CORE_NUM; i++)
		free_irq(p_cur_core[i].irq_no, NULL);
	iounmap(pdpureg);

	// remove extend mdoules
	softmax_remove();

	return 0;
}

static const struct of_device_id dpu_dt_ids[] = { { .compatible = "deephi, dpu" },
						{ .compatible = "deephi,dpu" },
						{ .compatible = "xilinx, dpu" },
						{ .compatible = "xilinx,dpu" },
						{ /* end of table */ } };

static struct platform_driver dpu_drv = {
	.driver = {
		.name = "dpu",
		.of_match_table = dpu_dt_ids,
	},
	.probe = dpu_probe,
	.remove = dpu_remove,
};

/**
 * dpu initialize function
 */
static int __init dpu_init(void)
{
	return platform_driver_register(&dpu_drv);
}

/**
 * dpu uninstall function
 */
static void __exit dpu_exit(void)
{
	platform_driver_unregister(&dpu_drv);
}

//----------------------------------------------------------
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Xilinx");
module_init(dpu_init);
module_exit(dpu_exit);
