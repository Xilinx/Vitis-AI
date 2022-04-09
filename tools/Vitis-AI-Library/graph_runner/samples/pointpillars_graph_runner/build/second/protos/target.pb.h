// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: second/protos/target.proto

#ifndef PROTOBUF_second_2fprotos_2ftarget_2eproto__INCLUDED
#define PROTOBUF_second_2fprotos_2ftarget_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3004000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3004000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "second/protos/anchors.pb.h"
#include "second/protos/similarity.pb.h"
// @@protoc_insertion_point(includes)
namespace second {
namespace protos {
class TargetAssigner;
class TargetAssignerDefaultTypeInternal;
extern TargetAssignerDefaultTypeInternal _TargetAssigner_default_instance_;
}  // namespace protos
}  // namespace second

namespace second {
namespace protos {

namespace protobuf_second_2fprotos_2ftarget_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[];
  static const ::google::protobuf::uint32 offsets[];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static void InitDefaultsImpl();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_second_2fprotos_2ftarget_2eproto

// ===================================================================

class TargetAssigner : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:second.protos.TargetAssigner) */ {
 public:
  TargetAssigner();
  virtual ~TargetAssigner();

  TargetAssigner(const TargetAssigner& from);

  inline TargetAssigner& operator=(const TargetAssigner& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  TargetAssigner(TargetAssigner&& from) noexcept
    : TargetAssigner() {
    *this = ::std::move(from);
  }

  inline TargetAssigner& operator=(TargetAssigner&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const TargetAssigner& default_instance();

  static inline const TargetAssigner* internal_default_instance() {
    return reinterpret_cast<const TargetAssigner*>(
               &_TargetAssigner_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(TargetAssigner* other);
  friend void swap(TargetAssigner& a, TargetAssigner& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline TargetAssigner* New() const PROTOBUF_FINAL { return New(NULL); }

  TargetAssigner* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const TargetAssigner& from);
  void MergeFrom(const TargetAssigner& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(TargetAssigner* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .second.protos.AnchorGeneratorCollection anchor_generators = 1;
  int anchor_generators_size() const;
  void clear_anchor_generators();
  static const int kAnchorGeneratorsFieldNumber = 1;
  const ::second::protos::AnchorGeneratorCollection& anchor_generators(int index) const;
  ::second::protos::AnchorGeneratorCollection* mutable_anchor_generators(int index);
  ::second::protos::AnchorGeneratorCollection* add_anchor_generators();
  ::google::protobuf::RepeatedPtrField< ::second::protos::AnchorGeneratorCollection >*
      mutable_anchor_generators();
  const ::google::protobuf::RepeatedPtrField< ::second::protos::AnchorGeneratorCollection >&
      anchor_generators() const;

  // string class_name = 5;
  void clear_class_name();
  static const int kClassNameFieldNumber = 5;
  const ::std::string& class_name() const;
  void set_class_name(const ::std::string& value);
  #if LANG_CXX11
  void set_class_name(::std::string&& value);
  #endif
  void set_class_name(const char* value);
  void set_class_name(const char* value, size_t size);
  ::std::string* mutable_class_name();
  ::std::string* release_class_name();
  void set_allocated_class_name(::std::string* class_name);

  // .second.protos.RegionSimilarityCalculator region_similarity_calculator = 6;
  bool has_region_similarity_calculator() const;
  void clear_region_similarity_calculator();
  static const int kRegionSimilarityCalculatorFieldNumber = 6;
  const ::second::protos::RegionSimilarityCalculator& region_similarity_calculator() const;
  ::second::protos::RegionSimilarityCalculator* mutable_region_similarity_calculator();
  ::second::protos::RegionSimilarityCalculator* release_region_similarity_calculator();
  void set_allocated_region_similarity_calculator(::second::protos::RegionSimilarityCalculator* region_similarity_calculator);

  // float sample_positive_fraction = 2;
  void clear_sample_positive_fraction();
  static const int kSamplePositiveFractionFieldNumber = 2;
  float sample_positive_fraction() const;
  void set_sample_positive_fraction(float value);

  // uint32 sample_size = 3;
  void clear_sample_size();
  static const int kSampleSizeFieldNumber = 3;
  ::google::protobuf::uint32 sample_size() const;
  void set_sample_size(::google::protobuf::uint32 value);

  // bool use_rotate_iou = 4;
  void clear_use_rotate_iou();
  static const int kUseRotateIouFieldNumber = 4;
  bool use_rotate_iou() const;
  void set_use_rotate_iou(bool value);

  // @@protoc_insertion_point(class_scope:second.protos.TargetAssigner)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedPtrField< ::second::protos::AnchorGeneratorCollection > anchor_generators_;
  ::google::protobuf::internal::ArenaStringPtr class_name_;
  ::second::protos::RegionSimilarityCalculator* region_similarity_calculator_;
  float sample_positive_fraction_;
  ::google::protobuf::uint32 sample_size_;
  bool use_rotate_iou_;
  mutable int _cached_size_;
  friend struct protobuf_second_2fprotos_2ftarget_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// TargetAssigner

// repeated .second.protos.AnchorGeneratorCollection anchor_generators = 1;
inline int TargetAssigner::anchor_generators_size() const {
  return anchor_generators_.size();
}
inline void TargetAssigner::clear_anchor_generators() {
  anchor_generators_.Clear();
}
inline const ::second::protos::AnchorGeneratorCollection& TargetAssigner::anchor_generators(int index) const {
  // @@protoc_insertion_point(field_get:second.protos.TargetAssigner.anchor_generators)
  return anchor_generators_.Get(index);
}
inline ::second::protos::AnchorGeneratorCollection* TargetAssigner::mutable_anchor_generators(int index) {
  // @@protoc_insertion_point(field_mutable:second.protos.TargetAssigner.anchor_generators)
  return anchor_generators_.Mutable(index);
}
inline ::second::protos::AnchorGeneratorCollection* TargetAssigner::add_anchor_generators() {
  // @@protoc_insertion_point(field_add:second.protos.TargetAssigner.anchor_generators)
  return anchor_generators_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::second::protos::AnchorGeneratorCollection >*
TargetAssigner::mutable_anchor_generators() {
  // @@protoc_insertion_point(field_mutable_list:second.protos.TargetAssigner.anchor_generators)
  return &anchor_generators_;
}
inline const ::google::protobuf::RepeatedPtrField< ::second::protos::AnchorGeneratorCollection >&
TargetAssigner::anchor_generators() const {
  // @@protoc_insertion_point(field_list:second.protos.TargetAssigner.anchor_generators)
  return anchor_generators_;
}

// float sample_positive_fraction = 2;
inline void TargetAssigner::clear_sample_positive_fraction() {
  sample_positive_fraction_ = 0;
}
inline float TargetAssigner::sample_positive_fraction() const {
  // @@protoc_insertion_point(field_get:second.protos.TargetAssigner.sample_positive_fraction)
  return sample_positive_fraction_;
}
inline void TargetAssigner::set_sample_positive_fraction(float value) {
  
  sample_positive_fraction_ = value;
  // @@protoc_insertion_point(field_set:second.protos.TargetAssigner.sample_positive_fraction)
}

// uint32 sample_size = 3;
inline void TargetAssigner::clear_sample_size() {
  sample_size_ = 0u;
}
inline ::google::protobuf::uint32 TargetAssigner::sample_size() const {
  // @@protoc_insertion_point(field_get:second.protos.TargetAssigner.sample_size)
  return sample_size_;
}
inline void TargetAssigner::set_sample_size(::google::protobuf::uint32 value) {
  
  sample_size_ = value;
  // @@protoc_insertion_point(field_set:second.protos.TargetAssigner.sample_size)
}

// bool use_rotate_iou = 4;
inline void TargetAssigner::clear_use_rotate_iou() {
  use_rotate_iou_ = false;
}
inline bool TargetAssigner::use_rotate_iou() const {
  // @@protoc_insertion_point(field_get:second.protos.TargetAssigner.use_rotate_iou)
  return use_rotate_iou_;
}
inline void TargetAssigner::set_use_rotate_iou(bool value) {
  
  use_rotate_iou_ = value;
  // @@protoc_insertion_point(field_set:second.protos.TargetAssigner.use_rotate_iou)
}

// string class_name = 5;
inline void TargetAssigner::clear_class_name() {
  class_name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& TargetAssigner::class_name() const {
  // @@protoc_insertion_point(field_get:second.protos.TargetAssigner.class_name)
  return class_name_.GetNoArena();
}
inline void TargetAssigner::set_class_name(const ::std::string& value) {
  
  class_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:second.protos.TargetAssigner.class_name)
}
#if LANG_CXX11
inline void TargetAssigner::set_class_name(::std::string&& value) {
  
  class_name_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:second.protos.TargetAssigner.class_name)
}
#endif
inline void TargetAssigner::set_class_name(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  class_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:second.protos.TargetAssigner.class_name)
}
inline void TargetAssigner::set_class_name(const char* value, size_t size) {
  
  class_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:second.protos.TargetAssigner.class_name)
}
inline ::std::string* TargetAssigner::mutable_class_name() {
  
  // @@protoc_insertion_point(field_mutable:second.protos.TargetAssigner.class_name)
  return class_name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* TargetAssigner::release_class_name() {
  // @@protoc_insertion_point(field_release:second.protos.TargetAssigner.class_name)
  
  return class_name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void TargetAssigner::set_allocated_class_name(::std::string* class_name) {
  if (class_name != NULL) {
    
  } else {
    
  }
  class_name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), class_name);
  // @@protoc_insertion_point(field_set_allocated:second.protos.TargetAssigner.class_name)
}

// .second.protos.RegionSimilarityCalculator region_similarity_calculator = 6;
inline bool TargetAssigner::has_region_similarity_calculator() const {
  return this != internal_default_instance() && region_similarity_calculator_ != NULL;
}
inline void TargetAssigner::clear_region_similarity_calculator() {
  if (GetArenaNoVirtual() == NULL && region_similarity_calculator_ != NULL) delete region_similarity_calculator_;
  region_similarity_calculator_ = NULL;
}
inline const ::second::protos::RegionSimilarityCalculator& TargetAssigner::region_similarity_calculator() const {
  const ::second::protos::RegionSimilarityCalculator* p = region_similarity_calculator_;
  // @@protoc_insertion_point(field_get:second.protos.TargetAssigner.region_similarity_calculator)
  return p != NULL ? *p : *reinterpret_cast<const ::second::protos::RegionSimilarityCalculator*>(
      &::second::protos::_RegionSimilarityCalculator_default_instance_);
}
inline ::second::protos::RegionSimilarityCalculator* TargetAssigner::mutable_region_similarity_calculator() {
  
  if (region_similarity_calculator_ == NULL) {
    region_similarity_calculator_ = new ::second::protos::RegionSimilarityCalculator;
  }
  // @@protoc_insertion_point(field_mutable:second.protos.TargetAssigner.region_similarity_calculator)
  return region_similarity_calculator_;
}
inline ::second::protos::RegionSimilarityCalculator* TargetAssigner::release_region_similarity_calculator() {
  // @@protoc_insertion_point(field_release:second.protos.TargetAssigner.region_similarity_calculator)
  
  ::second::protos::RegionSimilarityCalculator* temp = region_similarity_calculator_;
  region_similarity_calculator_ = NULL;
  return temp;
}
inline void TargetAssigner::set_allocated_region_similarity_calculator(::second::protos::RegionSimilarityCalculator* region_similarity_calculator) {
  delete region_similarity_calculator_;
  region_similarity_calculator_ = region_similarity_calculator;
  if (region_similarity_calculator) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_set_allocated:second.protos.TargetAssigner.region_similarity_calculator)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)


}  // namespace protos
}  // namespace second

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_second_2fprotos_2ftarget_2eproto__INCLUDED