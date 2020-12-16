/*
 * Copyright 2019 Xilinx Inc.
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
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>
#include <xir/util/data_type.hpp>

#ifdef __QNX__
#include <sys/elf.h>
#else
#include <elf.h>
#endif

#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
template <typename T>
struct env_config_helper {
  static inline T from_string(const char* s);
};

template <typename T, typename env_name>
struct env_config {
  static T init() {
    const char* name = env_name::get_name();
    const char* defvalue = env_name::get_default_value();
    const char* p = getenv(name);
    const char* pstr = p != nullptr ? p : defvalue;
    const T value = env_config_helper<T>::from_string(pstr);
    return value;
  }
  static T value;
};
template <typename T, typename env_name>
T env_config<T, env_name>::value = env_config<T, env_name>::init();

template <>
inline int env_config_helper<int>::from_string(const char* s) {
  return std::stoi(std::string(s));
}
template <>
inline std::string env_config_helper<std::string>::from_string(const char* s) {
  return std::string(s);
}

#define DEF_ENV_PARAM_2(param_name, defvalue1, type)                           \
  struct ENV_PARAM_##param_name                                                \
      : public env_config<type, ENV_PARAM_##param_name> {                      \
    static const char* get_name() { return #param_name; }                      \
    static const char* get_default_value() { return defvalue1; }               \
  };

#define ENV_PARAM(param_name) (ENV_PARAM_##param_name::value)

#define DEF_ENV_PARAM(param_name, defvalue1)                                   \
  DEF_ENV_PARAM_2(param_name, defvalue1, int)

DEF_ENV_PARAM(DEBUG_ELF_XIR, "0")
DEF_ENV_PARAM(DEBUG_ELF, "0")
// this struct is defined by elf format, try to be compatible
namespace {
struct tensor {
  uint32_t attr;
  uint32_t height;
  uint32_t width;
  uint32_t channel;
  uint32_t addr_logical;
  uint32_t size;
  uint32_t fix_width;
  int32_t fix_pos;
  uint32_t channel_stride;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
};

class buffer_object_fd {
 public:
  static std::shared_ptr<buffer_object_fd> create(const std::string& name,
                                                  int flags);

 public:
  explicit buffer_object_fd(const std::string& name, int flags);
  buffer_object_fd(const buffer_object_fd&) = delete;
  buffer_object_fd& operator=(const buffer_object_fd& other) = delete;
  virtual ~buffer_object_fd();

  int fd() { return fd_; }

 private:
  int fd_;
};

#define ELF_PHDR(name)                                                         \
  (is_class32() ? get<Elf32_Ehdr>(0)->name : get<Elf64_Ehdr>(0)->name)

#define ELF_SHDR(i, name)                                                      \
  (is_class32() ? get<Elf32_Shdr>(shoff())[i].name                             \
                : get<Elf64_Shdr>(shoff())[i].name)

class Elf {
 public:
  static std::unique_ptr<Elf> create(const std::string& filename,
                                     const std::string& kernel_name);

  static std::vector<std::pair<std::string, std::unique_ptr<Elf>>> create(
      const std::string& filename);

 public:
  Elf(const std::string& filename, const std::string& kernel_name);
  ~Elf();

 public:
  std::string Show();
  size_t CodeSize();
  int8_t* Code();
  size_t InitialCodeSize();
  int8_t* InitialCode();
  size_t ParameterSize();
  int8_t* Parameter();
  int8_t* Symtab();
  struct node;
  const std::vector<node*>& Nodes() const { return nodes_; }
  node* FindNodeByName(const char* name);
  tensor* FindInputTensorByName(const char* name, int idx);
  tensor* FindOutputTensorByName(const char* name, int idx);
  const std::vector<std::string>& kernels() const { return kernels_; }

 public:
  struct metadata {
    uint32_t dpu_arch;           //
    uint32_t ver_dnnc;           // =  1;
    uint32_t mode;               // =  2;
    uint32_t node_cnt;           // =  3;
    uint32_t tensor_size;        // =  4;
    uint32_t kernel_io_size;     // =  5;
    uint32_t kernel_mean_c1;     // =  6;
    uint32_t kernel_mean_c2;     // =  7;
    uint32_t kernel_mean_c3;     // =  8;
    uint16_t abi_ver_minor;      // = // LSB(0~15): minor verion
    uint16_t abi_ver_major;      // = // MSB(16~31): major version
    uint16_t dpu_ver_target;     // = 10;  // LSB(0~15): dpu target
    uint16_t dpu_ver_arch_type;  // = 10;  // MSB(16~31): dpu arch type
    uint32_t tensor_cnt;         // =  11;
  };
  template <typename T>
  struct list {
    uint32_t cnt;
    T data[];
    template <typename NextType>
    NextType* next() {
      return reinterpret_cast<NextType*>(reinterpret_cast<char*>(this) +
                                         cnt * sizeof(T) + sizeof(cnt));
    }
  };
  struct node {
    uint32_t type;
    uint32_t name;
    uint64_t workload;
    struct reg {
      uint16_t reg_id;
      uint16_t data_type;
    };
    uint32_t* marker[];
    list<reg>* regs() { return reinterpret_cast<list<reg>*>(&marker[0]); };
    list<uint32_t>* inputs() { return regs()->next<list<uint32_t>>(); }
    list<uint32_t>* outputs() { return inputs()->next<list<uint32_t>>(); };
    struct code {
      uint32_t offset;
      uint32_t size;
      uint32_t name_idx;
      uint32_t align;
    };
    list<code>* codes() { return outputs()->next<list<code>>(); };
    struct param {
      uint32_t offset;
      uint32_t size;
      uint32_t fix_w;
      uint32_t fix_p;
      uint32_t name_idx;
      uint32_t align;
      uint32_t height;
      uint32_t width;
      uint32_t channel;
      uint32_t out_channel;
    };
    list<param>* param() {  //
      return codes()->next<list<struct param>>();
    }
    list<uint32_t>* pre_nodes() { return param()->next<list<uint32_t>>(); }
    list<uint32_t>* suc_nodes() { return pre_nodes()->next<list<uint32_t>>(); };
  };
  // for internal use, but still public
  template <typename T>
  T* get(size_t offset) {
    return reinterpret_cast<T*>(static_cast<char*>(data_) + offset);
  }
  bool is_class32() {
    return get<Elf32_Ehdr>(0)->e_ident[EI_CLASS] == ELFCLASS32;
  }
  bool is_class64() {
    return get<Elf32_Ehdr>(0)->e_ident[EI_CLASS] == ELFCLASS64;
  }

  size_t shstrndx() { return ELF_PHDR(e_shstrndx); }
  size_t shoff() { return ELF_PHDR(e_shoff); }

  size_t shnum() { return ELF_PHDR(e_shnum); }

  std::string deephi_string(size_t offset) {
    return std::string(
        get<char>(section_offset(section_deephi_strtab_) + offset));
  }

  std::string strtab_string(size_t offset) {
    return std::string(get<char>(section_offset(section_strtab_) + offset));
  }

  std::string section_name(int section_idx) {
    return std::string(get<char>(ELF_SHDR(shstrndx(), sh_offset) +
                                 ELF_SHDR(section_idx, sh_name)));
  }

  size_t section_offset(int section_idx) {
    return ELF_SHDR(section_idx, sh_offset);
  }

  size_t section_size(int section_idx) {
    return ELF_SHDR(section_idx, sh_size);
  }

  size_t section_link(int section_idx) {
    return ELF_SHDR(section_idx, sh_link);
  }

  size_t section_entry_size(int section_idx) {
    return ELF_SHDR(section_idx, sh_entsize);
  }

  metadata* get_metadata() {
    return get<metadata>(section_offset(section_metadata_));
  }

  node* get_node() {  //
    return get<node>(section_offset(section_node_));
  }
  std::string get_file_name() { return filename_; }
  std::string get_kernel_name() { return kernel_; }
  std::string dump_meta_data();
  std::string dump_tensor(tensor* t);
  void parse_sections();
  void parse_section(size_t i);
  std::unique_ptr<std::pair<uint32_t, uint32_t>> get_preload_offset_and_size(
      const std::string& layer);
  size_t get_num_of_tensors() {  //
    return section_size(section_tensor_) / sizeof(tensor);
  }
  tensor* get_tensor(int i) {
    return get<tensor>(section_offset(section_tensor_) + i * sizeof(tensor));
  }
  void parse_nodes();
  void parse_symtab();
  std::string dump_nodes();
  std::string dump_node(node* n);

 private:
  std::string filename_;
  std::shared_ptr<buffer_object_fd> fd_;
  size_t size_;
  void* data_;
  std::string kernel_;
  uint32_t section_metadata_;
  uint32_t section_strtab_;
  uint32_t section_deephi_strtab_;
  uint32_t section_node_;
  uint32_t section_parameter_;
  uint32_t section_code_;
  uint32_t section_initial_code_;
  uint32_t section_tensor_;
  uint32_t section_symtab_;
  std::vector<node*> nodes_;
  std::vector<std::string> kernels_;
};

class ElfSectionBuilder;
struct ElfBuilder {
  ElfBuilder(std::ostream& out) : out_{out} {};
  void build();
  ElfSectionBuilder* new_section(const std::string& name);

 private:
  ElfSectionBuilder* get_section(const std::string& name);
  ElfSectionBuilder* get_or_new_section(const std::string& name);
  void prepare_build();
  void build_header();
  void build_section_headers();
  void build_section_header(ElfSectionBuilder* s);
  void build_sections();
  void build_section(ElfSectionBuilder* s);
  uint32_t total_section_size();

 private:
  std::ostream& out_;
  std::vector<std::unique_ptr<ElfSectionBuilder>> sections_;
};

struct ElfSectionBuilder {
 public:
  ElfSectionBuilder(const std::string& name, uint32_t id);
  uint32_t allocate_string(const std::string& str);
  void allocate_section_name(ElfSectionBuilder* str);
  uint32_t allocate_section_space(uint32_t pos);
  const std::string& get_name() const { return name_; }
  uint32_t get_id() const { return id_; }
  std::string data() const { return out_.str(); }
  size_t size() { return out_.tellp(); }
  uint32_t get_section_name();
  uint32_t offset();
  void align();
  void set_type(uint32_t type) { type_ = type; }
  uint32_t get_type() const { return type_; }
  template <typename X>
  void write(const X& x) {
    CHECK(out_.write(reinterpret_cast<const char*>(&x), sizeof(x)).good());
  }
  void write(const std::vector<char>& x) {
    CHECK(out_.write(&x[0], x.size()).good());
  }
  void write(const std::string& x) {
    CHECK(out_.write(&x[0], x.size()).good());
  }

 private:
  std::string name_;
  uint32_t id_;
  std::ostringstream out_;
  uint32_t section_name_;
  uint32_t offset_;
  uint32_t type_;
};

struct DnncKernel {
 public:
  DnncKernel(const std::string& name, ElfBuilder* elf);

 public:
  void set_kernel_iosize(size_t size) { kernel_io_size_ = size; }
  void set_num_of_nodes(size_t size) { num_of_nodes_ = size; }
  void build_meta_section();
  void build_code_section(const std::vector<char>& code);
  void build_parameter_section(const std::vector<char>& parameter);
  void build_node_section();
  void build_tensor_section();
  void add_tensor(const tensor& tensor) { tensors_.emplace_back(tensor); };
  void add_node(const std::function<void(std::ostream&)>& builder);
  uint32_t deephi_allocate_string(const std::string& str) {
    return strtab_->allocate_string(str);
  }
  uint32_t find_tensor_by_ddr_addr(uint32_t ddr_addr);

 private:
  ElfSectionBuilder* strtab_;
  ElfSectionBuilder* metadata_;
  ElfSectionBuilder* configurable_;
  ElfSectionBuilder* tensor_;
  ElfSectionBuilder* node_;
  ElfSectionBuilder* parameter_;
  ElfSectionBuilder* code_;
  std::string name_;
  size_t kernel_io_size_ = 0u;
  std::vector<tensor> tensors_ = {};
  size_t num_of_nodes_ = 0u;
};
template <typename T>
static inline std::ostream& operator<<(std::ostream& out,
                                       const std::vector<T>& v) {
  int c = 0;
  out << "[";
  for (const auto x : v) {
    if (c++ != 0) {
      out << ",";
    }
    out << x;
  }
  out << "]";
  return out;
}

using namespace std;

template <typename K, typename T>
struct WeakStore {
  template <typename... Args>
  static std::shared_ptr<T> create(const K& key, Args&&... args) {
    static std::unordered_map<K, std::weak_ptr<T>> the_store_;
    std::shared_ptr<T> ret;
    if (the_store_[key].expired()) {
      ret = std::make_shared<T>(std::forward<Args>(args)...);
      the_store_[key] = ret;
    }
    ret = the_store_[key].lock();
    assert(ret != nullptr);
    return ret;
  }
};

static size_t get_size(int fd) {
  struct stat statbuf;
  const auto r_stat = fstat(fd, &statbuf);
  CHECK_EQ(r_stat, 0) << "fstat error: ";
  CHECK_GT(statbuf.st_size, 0) << "must not empty file";
  return statbuf.st_size;
}
static void* my_map(int fd, size_t size) {
  auto p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  CHECK_NE(p, MAP_FAILED) << "cannot mmap";
  return p;
}

std::shared_ptr<buffer_object_fd> buffer_object_fd::create(
    const std::string& name, int flags) {
  return WeakStore<std::string, buffer_object_fd>::create(name, name, flags);
}

static int my_open(const std::string& name, int flags) {
  auto fd = open(name.c_str(), flags);
  CHECK_GT(fd, 0) << ", open(" << name << ") failed.";
  return fd;
}

buffer_object_fd::buffer_object_fd(const std::string& name, int flags)
    : fd_{my_open(name, flags)} {}
buffer_object_fd::~buffer_object_fd() { close(fd_); }

std::unique_ptr<Elf> Elf::create(const std::string& filename,
                                 const std::string& kernel_name) {
  return std::unique_ptr<Elf>(new Elf(filename, kernel_name));
}

std::vector<std::pair<std::string, std::unique_ptr<Elf>>> Elf::create(
    const std::string& filename) {
  auto elf = Elf::create(filename, "");
  auto ret = std::vector<std::pair<std::string, std::unique_ptr<Elf>>>{};
  for (const auto& k : elf->kernels()) {
    ret.emplace_back(std::make_pair(k, Elf::create(filename, k)));
  }
  return ret;
}

static std::shared_ptr<buffer_object_fd> create_fd(
    const std::string& filename) {
  auto ret = buffer_object_fd::create(filename, O_RDONLY | O_CLOEXEC);
  CHECK_GT(ret->fd(), 0) << "cannot open filename";
  return ret;
};
Elf::Elf(const std::string& filename, const std::string& kernel_name)

    : filename_{filename},
      fd_{create_fd(filename)},
      size_{get_size(fd_->fd())},
      data_{my_map(fd_->fd(), size_)},
      kernel_{kernel_name},
      section_metadata_{0},
      section_strtab_{0},
      section_deephi_strtab_{0},
      section_node_{0},
      section_parameter_{0},
      section_code_{0},
      section_initial_code_{0},
      section_tensor_{0},
      section_symtab_{0},
      nodes_() {
  auto is_elf = get<Elf32_Ehdr>(0)->e_ident[EI_MAG0] == ELFMAG0 &&
                get<Elf32_Ehdr>(0)->e_ident[EI_MAG1] == ELFMAG1 &&
                get<Elf32_Ehdr>(0)->e_ident[EI_MAG2] == ELFMAG2 &&
                get<Elf32_Ehdr>(0)->e_ident[EI_MAG3] == ELFMAG3;
  CHECK(is_elf == true) << ", " << filename << " is not a elf file";
  CHECK(is_class32() || is_class64())
      << ", only class32 or class64 is supported";
  parse_sections();
}
Elf::~Elf() {
  munmap(data_, size_);
  LOG_IF(INFO, ENV_PARAM(DEBUG_ELF)) << "close elf file " << filename_;
}

int8_t* Elf::Code() { return get<int8_t>(section_offset(section_code_)); }

size_t Elf::ParameterSize() { return section_size(section_parameter_); }
int8_t* Elf::Parameter() {
  return get<int8_t>(section_offset(section_parameter_));
}

void Elf::parse_sections() {
  for (auto i = 1u; i < shnum(); ++i) {
    parse_section(i);
  }
  parse_nodes();
  parse_symtab();
}

void Elf::parse_section(size_t idx) {
  auto name = section_name(idx);
  // LOG(INFO) << "parsing section[" << idx << "]" << name;
  if (name == ".deephi.metadata." + kernel_) {
    section_metadata_ = idx;
  } else if (name == ".deephi.strtab." + kernel_) {
    section_deephi_strtab_ = idx;
  } else if (name == ".strtab") {
    section_strtab_ = idx;
  } else if (name == ".deephi.node." + kernel_) {
    section_node_ = idx;
  } else if (name == ".deephi.parameter." + kernel_) {
    section_parameter_ = idx;
  } else if (name == ".deephi.code." + kernel_) {
    section_code_ = idx;
  } else if (name == ".deephi.initial_code." + kernel_) {
    section_initial_code_ = idx;
  } else if (name == ".deephi.tensor." + kernel_) {
    section_tensor_ = idx;
  } else if (name == ".symtab") {
    section_symtab_ = idx;
  } else {
    /* DLOG(INFO) << "unknown section " << section_name(idx)
               << " filename = " << filename_;
    */
  }
  const char nn[] = ".deephi.metadata.";
  const size_t start = sizeof(nn) - 1;
  if (name.find(nn) == 0) {
    kernels_.emplace_back(name.substr(start));
  }
}

void Elf::parse_nodes() {
  if (section_node_ == 0) {
    // kernel name is not detected
    return;
  }
  nodes_.resize(get_metadata()->node_cnt);
  if (get_metadata()->node_cnt > 0) {
    auto idx = 0u;
    nodes_[0] = get<node>(section_offset(section_node_));
    for (idx = 1; idx < nodes_.size(); ++idx) {
      nodes_[idx] = nodes_[idx - 1]->suc_nodes()->next<node>();
    }
  }
}

std::unique_ptr<std::pair<uint32_t, uint32_t>> Elf::get_preload_offset_and_size(
    const std::string& layer) {
  std::unique_ptr<std::pair<uint32_t, uint32_t>> ret = nullptr;
  if (section_symtab_ == 0) {
    return ret;
  }
  auto size = section_size(section_symtab_);
  auto n_of_symbols = size / section_entry_size(section_symtab_);
  LOG_IF(INFO, ENV_PARAM(DEBUG_ELF))
      << "section_symtab_ " << section_symtab_ << " "  //
      << "    section_entry_size(section_symtab_) "
      << "sizeof(Elf64_Sym) " << sizeof(Elf64_Sym) << " "  //
      << "sizeof(Elf32_Sym) " << sizeof(Elf32_Sym) << " "  //
      << section_entry_size(section_symtab_) << " "        //
      << "n_of_symbols " << n_of_symbols << " "            //
      ;

#if defined __aarch64__
  auto symbols = get<Elf64_Sym>(section_offset(section_symtab_));
#elif defined __arm__
  auto symbols = get<Elf32_Sym>(section_offset(section_symtab_));
#elif defined __x86_64__
  auto symbols = get<Elf64_Sym>(section_offset(section_symtab_));
#elif defined __microblaze__
  // I am not sure it is 64bits or 32 bits to avoid recompile all
  // models, let's assume it is 64bits model
  auto symbols = get<Elf64_Sym>(section_offset(section_symtab_));
#else
#error "Platform not support!"
#endif
  std::string name =
      std::string("_dpu_") + kernel_ + "_" + layer + "_preload_code";
  for (auto i = 0u; i < n_of_symbols; ++i) {
    if (strtab_string(symbols[i].st_name) == name) {
      ret = std::unique_ptr<std::pair<uint32_t, uint32_t>>(
          new std::pair<uint32_t, uint32_t>(symbols[i].st_value,
                                            symbols[i].st_size));
      break;
    }
  }
  if (!ret && 0) {
    LOG(WARNING) << "cannot find symbol " << name;
  }
  return ret;
}

void Elf::parse_symtab() {
  if (section_symtab_ == 0) {
    return;
  }
  if (0) {
    auto size = section_size(section_symtab_);
    auto n_of_symbols = size / section_entry_size(section_symtab_);
    LOG(INFO) << "section_symtab_ " << section_symtab_ << " "  //
              << "    section_entry_size(section_symtab_) "
              << "sizeof(Elf64_Sym) " << sizeof(Elf64_Sym) << " "  //
              << "sizeof(Elf32_Sym) " << sizeof(Elf32_Sym) << " "  //
              << section_entry_size(section_symtab_) << " "        //
              << "n_of_symbols " << n_of_symbols << " "            //
        ;
    /*
#if defined __aarch64__
    auto symbols = get<Elf64_Sym>(section_offset(section_symtab_));
#elif defined __arm__
    auto symbols = get<Elf32_Sym>(section_offset(section_symtab_));
#elif defined __x86_64__
    auto symbols = get<Elf32_Sym>(section_offset(section_symtab_));
#else
#error "Platform not support!"
#endif
    */
#if defined __arm__
    auto symbols = get<Elf32_Sym>(section_offset(section_symtab_));
#else
    auto symbols = get<Elf64_Sym>(section_offset(section_symtab_));
#endif
    for (auto i = 0u; i < n_of_symbols; ++i) {
      LOG(INFO) << std::hex << "symbols[i].st_name "
                << strtab_string(symbols[i].st_name) << " "  //
                << std::dec << "symbols[i].st_value " << symbols[i].st_value
                << " "                                                   //
                << "symbols[i].st_size " << symbols[i].st_size << " "    //
                << "symbols[i].st_shndx " << symbols[i].st_shndx << " "  //
          ;
    }
    auto x = get_preload_offset_and_size("inception_b1_1x7_reduce");
    if (x) {
      LOG(INFO) << "x->first " << x->first << " "    //
                << "x->second " << x->second << " "  //
          ;
    }
  }
}

struct elf2xir {
  const char* DPU_OP_TYPE = "concat";
  const char* DPU_INPUT_OP = "concat";
  const std::string filename_;
  const std::string kernel_;
  std::vector<std::string> input_ops_name_;
  std::vector<std::string> output_ops_name_;
  Elf* elf_;
  Elf::metadata* meta_data_;
  xir::Op* previous_op_;
  xir::Op* last_op_;
  xir::Graph* graph_;
  std::map<int, int> tensor_id_2_node_id_;
  std::map<int, int> node_id_2_kernel_id_;
  std::map<int, std::vector<int>> tensor_id_2_input_tensor_ids_;
  std::map<int, bool> tensor_id_2_is_input_;
  std::map<int, xir::Op*> tensor_id_2_tf_node_;
  size_t io_size_ = 0;
  std::vector<int8_t> parameter_;
  elf2xir(Elf* elf)
      : filename_(elf->get_file_name()),
        kernel_(elf->get_kernel_name()),
        input_ops_name_{},
        output_ops_name_{},
        elf_{std::move(elf)},
        meta_data_{elf_->get_metadata()} {
    //    LOG(INFO) << "kernel name is " << kernel_;
    CHECK(!filename_.empty()) << "FILENAME = " << elf->get_file_name();
  }

 private:
  void add_metadata() {
    auto debug_mode = meta_data_->mode != 0;
    CHECK(!debug_mode) << "debug mode not supported yet.";
    auto io_size = meta_data_->kernel_io_size;
    io_size_ = io_size;
    auto parameter_size = elf_->ParameterSize();
    auto parameter = elf_->Parameter();
    parameter_.assign(parameter, parameter + parameter_size);
    return;
  }

  enum RtNodeType { DpuCodeNode = 1, DpuVirtNode = 2, CpuNode = 0x80 };
  enum RtTensorType {
    NormalTensor = (1 << 0),
    InputEdgeTensor = (1 << 1),
    OutputEdgeTensor = (1 << 2),
  };

  string tf_node_name(int tensor_id) {
    auto it_node_id = tensor_id_2_node_id_.find(tensor_id);
    auto found_node_id = it_node_id != tensor_id_2_node_id_.end();
    auto node_id = found_node_id ? it_node_id->second : -1;
    auto node_name = found_node_id
                         ? elf_->deephi_string(elf_->Nodes()[node_id]->name)
                         : kernel_ + string("_INPUT");
    return node_name;
  }

  string node_name(int node_id) {
    auto node_name = elf_->deephi_string(elf_->Nodes()[node_id]->name);
    return node_name;
  }
  std::unique_ptr<xir::Tensor> create_xir_tensor(const std::string& name,
                                                 const tensor* tensor) {
    CHECK(tensor != nullptr) << "not a tensor! name=" << name;
    auto batch = 1;
    auto height = (int)tensor->height;
    auto width = (int)tensor->width;
    auto channel = (int)tensor->channel;
    auto ret = xir::Tensor::create(name, {batch, height, width, channel},
                                   xir::DataType{xir::DataType::XINT, 8});
    auto reg_id = 1;  // REG ID is always 1 for now;
    auto ddr_addr = tensor->addr_logical;
    auto location = 1;  // always on ddr;
    auto fix_pos = tensor->fix_pos;
    ret->set_attr("reg_id", reg_id);
    ret->set_attr("ddr_addr", (std::int32_t)ddr_addr);
    ret->set_attr("location", location);
    ret->set_attr("fix_point", fix_pos);
    LOG_IF(INFO, ENV_PARAM(DEBUG_ELF_XIR))
        << "create tensor: "
        << "name " << name << " "                                          //
        << "ddr_addr " << std::hex << "0x" << ddr_addr << std::dec << " "  //
        << "fixpoint is " << fix_pos;
    return ret;
  }

  void connect_tf_nodes() {
    auto n_of_tensors = elf_->get_num_of_tensors();
    for (auto tensor_id = 0u; tensor_id < n_of_tensors; ++tensor_id) {
      auto it_node_id = tensor_id_2_node_id_.find(tensor_id);
      auto found_node_id = it_node_id != tensor_id_2_node_id_.end();
      if (found_node_id) {
        auto node_id = it_node_id->second;
        auto& input_tensor_ids = tensor_id_2_input_tensor_ids_[tensor_id];
        auto node = elf_->Nodes()[node_id];
        for (auto i = 0u; i < node->inputs()->cnt; ++i) {
          auto input_tensor_id = node->inputs()->data[i];
          input_tensor_ids.emplace_back(input_tensor_id);
        }
      }
    }
  }

  uint32_t get_4k_align_size(uint32_t code_size) {
    return (code_size & 0xfff) ? (((code_size >> 12) + 1) << 12) : code_size;
  }

  std::string code_buffer = "";
  // uint32_t node_code_offset = 0;
  // int preload_size = 0;

  std::map<int, int> build_tensor_id_2_node_id() {
    int node_id = 0;
    const auto& nodes = elf_->Nodes();
    auto ret = std::map<int, int>{};
    for (const auto& node : nodes) {
      for (auto i = 0u; i < node->outputs()->cnt; ++i) {
        auto tensor_id = node->outputs()->data[i];
        CHECK(ret.find(tensor_id) == ret.end())
            << " tensor id must belong to at most one node id. tensor_id = "
            << tensor_id << " , node_id=" << node_id;
        ret[tensor_id] = node_id;
      }
      node_id++;
    }
    return ret;
  }

  std::map<int, int> build_node_id_2_kernel_id() {
    auto ret = std::map<int, int>{};
    auto debug_mode = meta_data_->mode != 0;
    const auto& nodes = elf_->Nodes();
    int node_id = 0;
    int kernel_id = 1;
    for (const auto& node : nodes) {
      if (node->type == DpuCodeNode) {
        if (debug_mode) {
          ret[node_id] = kernel_id;
        } else {
          ret[node_id] = 0;  // all dpu kernel has the same kernel id
        }
      } else if (node->type == DpuVirtNode) {
        if (debug_mode) {
          ret[node_id] = kernel_id;
        } else {
          ret[node_id] = 0;  // all dpu kernel has the same kernel id
        }
      } else if (node->type == CpuNode) {
        ret[node_id] = kernel_id;
      } else {
        LOG(FATAL) << "unknown node type, id=" << node_id;
      }
      kernel_id++;
      node_id++;
    }
    return ret;
  }
  template <typename T>
  static std::string dump_object(const T& v) {
    std::ostringstream out;
    out << v;
    return out.str();
  }
  xir::Op* create_dpu_op(unsigned int tensor_id) {
    auto it_input_tensor_ids = tensor_id_2_input_tensor_ids_.find(tensor_id);
    CHECK(it_input_tensor_ids != tensor_id_2_input_tensor_ids_.end())
        << "cannot find input op! tensor_id=" << tensor_id;
    auto& input_tensor_ids = it_input_tensor_ids->second;
    auto input_ops = std::vector<xir::Op*>();
    for (auto idx = 0u; idx < input_tensor_ids.size(); ++idx) {
      auto input_tensor_id = input_tensor_ids[idx];
      auto it_input_op = tensor_id_2_tf_node_.find(input_tensor_id);
      CHECK(it_input_op != tensor_id_2_tf_node_.end())
          << "cannot find input op! tensor_id=" << tensor_id
          << ", input_tensor_id=" << input_tensor_id;
      auto op = it_input_op->second;
      input_ops.emplace_back(op);
    }
    //
    // dirty hack, all op are concat
    auto attrs = xir::Attrs::create();
    attrs->set_attr("axis", 0);
    auto tensor_op_name =
        kernel_ + std::string("_op_") + std::to_string(tensor_id);
    auto tensor = elf_->get_tensor(tensor_id);
    last_op_ = graph_->add_op(tensor_op_name, std::string(DPU_OP_TYPE),
                              std::move(attrs), {{"input", input_ops}});
    last_op_->replace_output_tensor(create_xir_tensor(
        tf_node_name(tensor_id) + "_" + std::to_string(tensor_id), tensor));
    return last_op_;
  }

  xir::Op* create_input_op(unsigned int tensor_id) {
    auto tensor_op_name =
        kernel_ + std::string("_op_") + std::to_string(tensor_id);
    auto tensor = elf_->get_tensor(tensor_id);
    auto attrs = xir::Attrs::create();
    attrs->set_attr("axis", 0);
    auto xir_tensor = create_xir_tensor(
        tf_node_name(tensor_id) + "_" + std::to_string(tensor_id), tensor);
    auto ret = graph_->add_op(tensor_op_name, std::string(DPU_INPUT_OP),
                              std::move(attrs), {{"input", {previous_op_}}});
    ret->replace_output_tensor(std::move(xir_tensor));
    return ret;
  }

  void add_tf_nodes() {
    auto n_of_tensors = elf_->get_num_of_tensors();
    LOG_IF(INFO, ENV_PARAM(DEBUG_ELF_XIR))
        << " filename = " << filename_ << " "
        << " kernel_name=" << elf_->get_kernel_name()
        << " n_of_tensors=" << n_of_tensors;
    for (auto tensor_id = 0u; tensor_id < n_of_tensors; ++tensor_id) {
      auto node_name = tf_node_name(tensor_id);
      auto it_node_id = tensor_id_2_node_id_.find(tensor_id);
      auto found_node_id = it_node_id != tensor_id_2_node_id_.end();
      auto node_id = found_node_id ? it_node_id->second : -1;
      auto it_kernel_id = found_node_id ? node_id_2_kernel_id_.find(node_id)
                                        : node_id_2_kernel_id_.end();
      auto found_kernel_id = it_kernel_id != node_id_2_kernel_id_.end();
      auto kernel_id = found_kernel_id ? it_kernel_id->second : -1;
      auto tensor = elf_->get_tensor(tensor_id);
      auto is_input = false;
      auto is_output = false;
      auto op_type = std::string(DPU_OP_TYPE);
      if (tensor->attr == NormalTensor) {
        is_input = false;
        is_output = false;
        op_type = std::string(DPU_OP_TYPE);
      } else if (tensor->attr == InputEdgeTensor) {
        is_input = true;
        is_output = false;
        op_type = std::string(DPU_INPUT_OP) + "_input";  // ugly code
      } else if (tensor->attr == OutputEdgeTensor) {
        is_input = false;
        is_output = true;
        op_type = std::string(DPU_OP_TYPE);
      } else {
        LOG(FATAL) << "Unkonwn tensor attr tensor_id =" << tensor_id;
      }

      auto tf_node = op_type == std::string(DPU_OP_TYPE)
                         ? create_dpu_op(tensor_id)
                         : create_input_op(tensor_id);
      tensor_id_2_tf_node_[tensor_id] = tf_node;

      if (found_node_id) {
        auto elf_node = elf_->Nodes()[node_id];
        if (elf_node->type == DpuCodeNode) {
        } else if (elf_node->type == DpuVirtNode) {
        } else if (elf_node->type == CpuNode) {
          LOG(FATAL) << "unsupported node type, id=" << node_id;
        } else {
          LOG(FATAL) << "unknown node type, id=" << node_id;
        }
      } else {
        CHECK_EQ(tf_node->get_type(), DPU_INPUT_OP);
      }
      tf_node->set_attr("is_graph_input", is_input);
      tensor_id_2_is_input_[tensor_id] = is_input;
      tf_node->set_attr("is_graph_output", is_output);
      tf_node->set_attr("fix_point", tensor->fix_pos);
      tf_node->set_attr(
          "super_layer_id",
          kernel_ + std::string("_") + (found_node_id ? node_name : "Input"));
      tf_node->set_attr("kernel_id",
                        kernel_ + std::string("_") + std::to_string(kernel_id));
      if (kernel_id == -1) {
        tf_node->set_attr<string>("node_type", "input");
      } else {
        tf_node->set_attr<string>("node_type", "node");
      }
      tf_node->set_attr("node_id", node_id);
      tf_node->set_attr("tensor_id", tensor_id);
      if (is_input) {
        input_ops_name_.emplace_back(tf_node->get_name());
      }
      if (is_output) {
        output_ops_name_.emplace_back(tf_node->get_name());
      }
    }
  }

  void process_kernel_subgraphs() {
    auto root = graph_->get_root_subgraph();
    // CHECK_EQ(root->get_children_num(), 2)
    //     << "only support one input kernel and one dpu kernel";
    for (auto& kernel_subgraph : root->get_children()) {
      auto device = kernel_subgraph->get_attr<std::string>("device");
      if (device == "DPU" && kernel_subgraph->get_name() == kernel_) {
        attach_code_to_kernel_subgraph(kernel_subgraph);
        attach_parameter_to_kernel_subgraph(kernel_subgraph);
        attach_input_and_output_ops_name_to_subgraph(kernel_subgraph);
        attach_runner_to_kernel_subgraph(kernel_subgraph);
      }
    }
  }
  void attach_code_to_kernel_subgraph(xir::Subgraph* kernel_subgraph) {
    std::vector<char> code_buffer;
    code_buffer.reserve(4096);
    auto code = elf_->Code();
    const auto& nodes = elf_->Nodes();
    auto node_id = 0;
    auto debug_mode = meta_data_->mode != 0;
    CHECK(!debug_mode)
        << "only release mode is supported. when code is attached to kernel "
           "subgraph";

    for (const auto& node : nodes) {
      auto node_name = elf_->deephi_string(node->name);
      if (node->type == DpuCodeNode) {
        CHECK_EQ(node->codes()->cnt, 1u)
            << "one DPU codde per super layer! node_name=" << node_name;
        auto code_size = node->codes()->data[0].size;
        auto code_offset = node->codes()->data[0].offset;
        code_buffer.insert(code_buffer.end(), code + code_offset,
                           code + code_offset + code_size);
      }
      node_id++;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_ELF_XIR))
        << "kernel =" << kernel_subgraph->get_name() << " kernel_=" << kernel_
        << " code size=" << code_buffer.size();
    kernel_subgraph->set_attr("mc_code", code_buffer);
  }

  void attach_parameter_to_kernel_subgraph(xir::Subgraph* kernel_subgraph) {
    auto parameter_size = elf_->ParameterSize();
    auto parameter = elf_->Parameter();
    kernel_subgraph->set_attr<std::map<std::string, std::vector<char>>>(
        "reg_id_to_parameter_value",
        {{"REG_0", {parameter, parameter + parameter_size}}});
  }
  void attach_runner_to_kernel_subgraph(xir::Subgraph* kernel_subgraph) {
    kernel_subgraph->set_attr<std::map<std::string, std::string>>(
        "runner", {{"run", "libvart-dpu-runner.so"}});
  }
  void attach_input_and_output_ops_name_to_subgraph(xir::Subgraph* subgraph) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_ELF_XIR))
        << "kernel=" << subgraph->get_name() << " "
        << "input_ops_name " << input_ops_name_ << " "
        << "output_ops_name_" << output_ops_name_ << " ";
    subgraph->set_attr("input_ops_name", input_ops_name_);
    subgraph->set_attr("output_ops_name", output_ops_name_);
    subgraph->set_attr<std::map<std::string, std::string>>(
        "reg_id_to_context_type", {{"REG_0", "CONST"}, {"REG_1", "DATA"}});
    subgraph->set_attr<std::map<std::string, int>>("reg_id_to_size",
                                                   {{"REG_1", (int)io_size_}});
    subgraph->set_attr<std::map<std::string, std::string>>(
        "reg_id_to_hw_segment", {{"REG_0", "W0"}, {"REG_1", "D0"}});
  }

 public:
  xir::Op* operator()(xir::Graph* g, xir::Op* op) {
    previous_op_ = op;
    graph_ = g;
    auto graph_name = filename_ + ":" + elf_->get_kernel_name();

    add_metadata();
    tensor_id_2_node_id_ = build_tensor_id_2_node_id();
    node_id_2_kernel_id_ = build_node_id_2_kernel_id();
    connect_tf_nodes();
    add_tf_nodes();
    return last_op_;
  }
  /* void add_subgraphs(xir::Graph* g) {
  // add_tf_nodes_preload();
  // add_super_layer_preload();
  add_subgraphs();
  return;
  }*/
  void process_kernel_subgraphs(xir::Graph* g) {
    process_kernel_subgraphs();
    return;
  }
};

}  // namespace
namespace xir {

void create_super_layer_subgraph(xir::Subgraph* kernel_subgraph) {
  kernel_subgraph->create_children();
  std::map<std::string, std::set<xir::Subgraph*>> superlayers;
  auto ops = kernel_subgraph->get_ops();
  for (auto op : ops) {
    auto super_layer = kernel_subgraph->find_op(op);
    auto super_layer_id = op->get_attr<std::string>("super_layer_id");
    superlayers[super_layer_id].emplace(super_layer);
  }
  for (auto& superlayer_id_and_subgraph : superlayers) {
    auto superlayer_subgraph =
        kernel_subgraph->merge_children(superlayer_id_and_subgraph.second);
    superlayer_subgraph->set_name(superlayer_id_and_subgraph.first);
  }
}
bool check_device_is_dpu(const std::string& kernel_id) {
  return kernel_id.substr(kernel_id.size() - 3) != "_-1" &&
         kernel_id != "global_kernel_id";
}

void add_subgraphs(xir::Graph* g) {
  //  auto debug_mode = meta_data_->mode != 0;
  auto ops = g->get_ops();
  auto root = g->get_root_subgraph();
  root->create_children();
  std::map<std::string, std::set<xir::Subgraph*>> kernel_subgraphs;
  for (auto op : ops) {
    auto subgraph = root->find_op(op);
    auto kernel_id = op->get_attr<std::string>("kernel_id");
    kernel_subgraphs[kernel_id].emplace(subgraph);
    // LOG(INFO) << "kernel id " << kernel_id;
  }
  // CHECK(!debug_mode) << "debug mode not supported yet.";
  // CHECK_EQ(kernel_subgraphs.size(), 2u)
  //     << "only support one input kernel and one dpu kernel.";
  for (auto& kernel_name_and_subgraphs : kernel_subgraphs) {
    auto kernel_subgraph =
        root->merge_children(kernel_name_and_subgraphs.second);
    auto kernel_id = kernel_name_and_subgraphs.first;
    if (check_device_is_dpu(kernel_id)) {
      kernel_subgraph->set_attr<std::string>("device", "DPU");
      // kernel_subgraph->set_name(kernel_id.substr(0, kernel_.size()));

      kernel_subgraph->set_name(
          kernel_id.substr(0, kernel_id.find_last_of("_")));
      // kernel_subgraph->set_name(elf_->get_kernel_name());
    } else {
      kernel_subgraph->set_attr<std::string>("device", "USER");
      kernel_subgraph->set_name(kernel_id);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_ELF_XIR))
        << "kernel_subgraphs.size() " << kernel_subgraphs.size()
        << " "  //
           " kernel-name= "
        << kernel_subgraph->get_name();
    create_super_layer_subgraph(kernel_subgraph);
  }
  return;
}

std::unique_ptr<xir::Graph> my_elf2xir(const std::string& filename) {
  auto graph_name = filename;
  auto graph = xir::Graph::create(graph_name);
  auto kernels = Elf::create(filename);
  auto op = graph->add_op("global_input", std::string("data"),
                          xir::Attrs::create(), {});
  op->replace_output_tensor(xir::Tensor::create(
      "global_input", {1, 1, 1, 1}, xir::DataType{xir::DataType::XINT, 8}));
  op->set_attr<string>("kernel_id", "global_kernel_id");
  op->set_attr<string>("node_type", "global");
  op->set_attr<string>("super_layer_id", "global_super_layer_id");
  auto objs = vector<elf2xir>{};
  if (0)
    for (auto& k : kernels) {
      LOG(INFO) << "k-> " << k.first << ":" << (void*)k.second.get();
    }
  for (auto& k : kernels) {
    auto elf2tf_obj = elf2xir(k.second.get());
    op = elf2tf_obj(graph.get(), op);
    objs.emplace_back(std::move(elf2tf_obj));
  }

  add_subgraphs(graph.get());
  for (auto& o : objs) {
    o.process_kernel_subgraphs(graph.get());
  }
  if (ENV_PARAM(DEBUG_ELF_XIR)) {
    graph->serialize("a.xmodel");
  }
  return graph;
}
}  // namespace xir

// Local Variables:
// mode:c++
// coding: utf-8-unix
// End:
