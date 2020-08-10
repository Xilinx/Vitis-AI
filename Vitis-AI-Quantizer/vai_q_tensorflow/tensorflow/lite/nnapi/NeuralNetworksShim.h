/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_NNAPI_NEURALNETWORKSSHIM_H_
#define TENSORFLOW_LITE_NNAPI_NEURALNETWORKSSHIM_H_

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"

// This interface is now deprecated. You should use instead
// nnapi_implementation.

// TODO(b/123017568): Update all current usages of this file.

// helpers

#define NNAPI_LOG(format, ...) fprintf(stderr, format "\n", __VA_ARGS__);
#define LOAD_FUNCTION(name) \
  static name##_fn fn = reinterpret_cast<name##_fn>(loadFunction(#name));
#define EXECUTE_FUNCTION(...) \
  if (fn != nullptr) {        \
    fn(__VA_ARGS__);          \
  }
#define EXECUTE_FUNCTION_RETURN(...) return fn != nullptr ? fn(__VA_ARGS__) : 0;

inline void* loadLibrary(const char* name) {
  // TODO: change RTLD_LOCAL? Assumes there can be multiple instances of nn
  // api RT
  void* handle = nullptr;
#ifdef __ANDROID__
  handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    NNAPI_LOG("nnapi error: unable to open library %s", name);
  }
#endif
  return handle;
}

// ASharedMemory_create was added in Android 8.0, so safe to use with NNAPI
// which was added in 8.1.
inline int ASharedMemory_create(const char* name, size_t size) {
  static void* handle = loadLibrary("libandroid.so");
  static ASharedMemory_create_fn fn =
      handle != nullptr ? reinterpret_cast<ASharedMemory_create_fn>(
                              dlsym(handle, "ASharedMemory_create"))
                        : nullptr;
  int fd = fn != nullptr ? fn(name, size) : -1;
  return fd;
}

inline void* getLibraryHandle() {
  static void* handle = loadLibrary("libneuralnetworks.so");
  return handle;
}

inline void* loadFunction(const char* name) {
  void* fn = nullptr;
  if (getLibraryHandle() != nullptr) {
    fn = dlsym(getLibraryHandle(), name);
  }
  if (fn == nullptr) {
    NNAPI_LOG("nnapi error: unable to open function %s", name);
  }
  return fn;
}

inline bool NNAPIExists() {
  static bool nnapi_is_available = getLibraryHandle();
  return nnapi_is_available;
}

// NN api types based on NNAPI header file
// https://developer.android.com/ndk/reference/group/neural-networks

/**
 * Creates a shared memory object from a file descriptor.
 *
 * The shared memory is backed by a file descriptor via mmap.
 * See {@link ANeuralNetworksMemory} for a description on how to use
 * this shared memory.
 *
 * @param size The requested size in bytes.
 *             Must not be larger than the file size.
 * @param prot The desired memory protection for the mapping.
 *             It is either PROT_NONE or the bitwise OR of one or
 *             more of the following flags: PROT_READ, PROT_WRITE.
 * @param fd The requested file descriptor.
 *           The file descriptor has to be mmap-able. The file
 *           descriptor will be duplicated.
 * @param offset The offset to the beginning of the file of the area to map.
 *               The offset has to be aligned to a page size.
 * @param memory The memory object to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 */
inline int ANeuralNetworksMemory_createFromFd(size_t size, int protect, int fd,
                                              size_t offset,
                                              ANeuralNetworksMemory** memory) {
  LOAD_FUNCTION(ANeuralNetworksMemory_createFromFd);
  EXECUTE_FUNCTION_RETURN(size, protect, fd, offset, memory);
}

/**
 * Delete a memory object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param memory The memory object to be freed.
 */
inline void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory) {
  LOAD_FUNCTION(ANeuralNetworksMemory_free);
  EXECUTE_FUNCTION(memory);
}

/**
 * Create an empty {@link ANeuralNetworksModel}.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuralNetworksExecution_startCompute} is invoked.
 *
 * The model should be constructed with calls to
 * {@link ANeuralNetworksModel_addOperation} and
 * {@link ANeuralNetworksModel_addOperand}
 *
 * <p>{@link ANeuralNetworksModel_finish} should be called once the model
 * has been fully constructed.</p>
 *
 * <p>{@link ANeuralNetworksModel_free} should be called once the model
 * is no longer needed.</p>
 *
 * @param model The {@link ANeuralNetworksModel} to be created.
 *              Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
  LOAD_FUNCTION(ANeuralNetworksModel_create);
  EXECUTE_FUNCTION_RETURN(model);
}

/**
 * Destroy a model.
 *
 * The model need not have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
inline void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
  LOAD_FUNCTION(ANeuralNetworksModel_free);
  EXECUTE_FUNCTION(model);
}

/**
 * Indicate that we have finished modifying a model. Required before
 * calling {@link ANeuralNetworksCompilation_compile}.
 *
 * An application is responsible to make sure that no other thread uses
 * the model at the same time.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be finished.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
  LOAD_FUNCTION(ANeuralNetworksModel_finish);
  EXECUTE_FUNCTION_RETURN(model);
}

/**
 * Add an operand to a model.
 *
 * The order in which the operands are added is important. The first one added
 * to a model will have the index value 0, the second 1, etc. These indexes are
 * used as operand identifiers in {@link ANeuralNetworksModel_addOperation},
 * {@link ANeuralNetworksExecution_setInput},
 * {@link ANeuralNetworksExecution_setInputFromMemory},
 * {@link ANeuralNetworksExecution_setOutput},
 * {@link ANeuralNetworksExecution_setOutputFromMemory} and
 * {@link ANeuralNetworksExecution_setOperandValue}.
 *
 * To build a model that can accommodate inputs of various sizes, as you may
 * want to do for a CNN, set the size of the dimensions that will vary at run
 * time to 0. If you do so, provide the full dimensions when calling
 * {@link ANeuralNetworksExecution_setInput} or {@link
 * ANeuralNetworksExecution_setInputFromMemory}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param type The {@link ANeuralNetworksOperandType} that describes the shape
 * of the operand.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_addOperand(
    ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type) {
  LOAD_FUNCTION(ANeuralNetworksModel_addOperand);
  EXECUTE_FUNCTION_RETURN(model, type);
}

/**
 * Sets an operand to a constant value.
 *
 * For scalar values, the content of buffer is copied into the model.
 *
 * For tensor values, a pointer to the buffer is stored within the model.
 * The application is responsible for not changing the content of this region
 * until all executions using this model have completed. As the data may
 * be copied during processing, modifying the data after this call yields
 * undefined results.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model,
                                                int32_t index,
                                                const void* buffer,
                                                size_t length) {
  LOAD_FUNCTION(ANeuralNetworksModel_setOperandValue);
  EXECUTE_FUNCTION_RETURN(model, index, buffer, length);
}

/**
 * Sets an operand's per channel quantization parameters.
 *
 * Sets parameters required by a tensor of type
 * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL}.
 * This function must be called for every tensor of type
 * {@link ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL} before
 * calling {@link ANeuralNetworksModel_finish}.
 *
 * Available since API level 29.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param channelQuant The per channel quantization parameters for the operand.
 *                    No memory in this struct needs to outlive the call to
 *                    this function.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksSymmPerChannelQuantParams* channelQuant) {
  LOAD_FUNCTION(ANeuralNetworksModel_setOperandSymmPerChannelQuantParams);
  EXECUTE_FUNCTION_RETURN(model, index, channelQuant);
}

/**
 * Sets an operand to a value stored in a memory object.
 *
 * The content of the memory is not copied. A reference to that memory is stored
 * inside the model. The application is responsible for not changing the content
 * of the memory region until all executions using this model have completed.
 * As the data may be copied during processing, modifying the data after this
 * call yields undefined results.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @param model The model to be modified.
 * @param index The index of the model operand we're setting.
 * @param buffer A pointer to the data to use.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_setOperandValueFromMemory(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksMemory* memory, size_t offset, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksModel_setOperandValueFromMemory);
  EXECUTE_FUNCTION_RETURN(model, index, memory, offset, length);
}

/**
 * Add an operation to a model.
 *
 * @param model The model to be modified.
 * @param type The type of the operation.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying each operand.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying each operand.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                             ANeuralNetworksOperationType type,
                                             uint32_t inputCount,
                                             const uint32_t* inputs,
                                             uint32_t outputCount,
                                             const uint32_t* outputs) {
  LOAD_FUNCTION(ANeuralNetworksModel_addOperation);
  EXECUTE_FUNCTION_RETURN(model, type, inputCount, inputs, outputCount,
                          outputs);
}

/**
 * Specifies which operands will be the model's inputs and outputs.
 *
 * An operand cannot be used for both input and output. Doing so will
 * return an error.
 *
 * @param model The model to be modified.
 * @param inputCount The number of entries in the inputs array.
 * @param inputs An array of indexes identifying the input operands.
 * @param outputCount The number of entries in the outputs array.
 * @param outputs An array of indexes identifying the output operands.
 *
 * The operands specified by inputs and outputs must have been
 * previously added by calls to {@link ANeuralNetworksModel_addOperand}.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 *
 */
inline int ANeuralNetworksModel_identifyInputsAndOutputs(
    ANeuralNetworksModel* model, uint32_t inputCount, const uint32_t* inputs,
    uint32_t outputCount, const uint32_t* outputs) {
  LOAD_FUNCTION(ANeuralNetworksModel_identifyInputsAndOutputs);
  EXECUTE_FUNCTION_RETURN(model, inputCount, inputs, outputCount, outputs);
}

/**
 * Specifies whether {@link ANEURALNETWORKS_TENSOR_FLOAT32} is allowed to be
 * calculated with range and/or precision as low as that of the IEEE 754 16-bit
 * floating-point format. By default, {@link ANEURALNETWORKS_TENSOR_FLOAT32}
 * must be calculated using at least the range and precision of the IEEE 754
 * 32-bit floating-point format.
 *
 * @param model The model to be modified.
 * @param allow 'true' indicates {@link ANEURALNETWORKS_TENSOR_FLOAT32} may be
 *              calculated with range and/or precision as low as that of the
 *              IEEE 754 16-bit floating point format. 'false' indicates
 *              {@link ANEURALNETWORKS_TENSOR_FLOAT32} must be calculated using
 *              at least the range and precision of the IEEE 754 32-bit floating
 *              point format.
 *
 * Attempting to modify a model once {@link ANeuralNetworksModel_finish} has
 * been called will return an error.
 *
 * Available since API level 28.
 *
 * See {@link ANeuralNetworksModel} for information on multithreaded usage.
 */
inline int ANeuralNetworksModel_relaxComputationFloat32toFloat16(
    ANeuralNetworksModel* model, bool allow) {
  LOAD_FUNCTION(ANeuralNetworksModel_relaxComputationFloat32toFloat16);
  EXECUTE_FUNCTION_RETURN(model, allow);
}

/**
 * Create a {@link ANeuralNetworksCompilation} to compile the given model.
 * This only creates the object. Compilation is only performed once
 * {@link ANeuralNetworksCompilation_start} is invoked.
 *
 * <p>The provided model must outlive the compilation.</p>
 *
 * The model must already have been finished by a call to
 * {@link ANeuralNetworksModel_finish}.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param model The {@link ANeuralNetworksModel} to be compiled.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the model is invalid.
 */
inline int ANeuralNetworksCompilation_create(
    ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_create);
  EXECUTE_FUNCTION_RETURN(model, compilation);
}

/**
 * Destroy a compilation.
 *
 * <p>If called on a compilation for which
 * {@link ANeuralNetworksCompilation_start} has been called, the
 * function will return immediately but will mark the compilation to be deleted
 * once the compilation completes. The {@link ANeuralNetworksCompilation_wait}
 * will return ERROR_DELETED.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param compilation The compilation to be destroyed. Passing NULL is
 * acceptable and results in no operation.
 */
inline void ANeuralNetworksCompilation_free(
    ANeuralNetworksCompilation* compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_free);
  EXECUTE_FUNCTION(compilation);
}

/**
 * Sets the execution preference.
 *
 * <p>Provides guidance to the runtime when trade-offs are possible.</p>
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param compilation The compilation to be modified.
 * @param preference Either {@link PREFER_LOW_POWER},
 *                  {@link PREFER_SINGLE_FAST_ANSWER}, or
 *                  {@link PREFER_SUSTAINED_SPEED}.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksCompilation_setPreference(
    ANeuralNetworksCompilation* compilation, int32_t preference) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_setPreference);
  EXECUTE_FUNCTION_RETURN(compilation, preference);
}

/**
 * Waits until the compilation completes.
 *
 * More than one thread can wait on a compilation. When the compilation
 * completes, all threads will be released.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the compilation completed normally.
 */
inline int ANeuralNetworksCompilation_finish(
    ANeuralNetworksCompilation* compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_finish);
  EXECUTE_FUNCTION_RETURN(compilation);
}
/**
 * Create a {@link ANeuralNetworksExecution} to apply the given compilation.
 * This only creates the object. Computation is only performed once
 * {@link ANeuralNetworksExecution_startCompute} is invoked.
 *
 * <p>The provided compilation must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param compilation The {@link ANeuralNetworksCompilation} to be evaluated.
 * @param execution The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the compilation is invalid.
 */
inline int ANeuralNetworksExecution_create(
    ANeuralNetworksCompilation* compilation,
    ANeuralNetworksExecution** execution) {
  LOAD_FUNCTION(ANeuralNetworksExecution_create);
  EXECUTE_FUNCTION_RETURN(compilation, execution);
}

/**
 * Destroy an execution.
 *
 * <p>If called on an execution for which
 * {@link ANeuralNetworksExecution_startCompute} has been called, the
 * function will return immediately but will mark the execution to be deleted
 * once the computation completes.   The {link ANeuralNetworksExecution_wait}
 * will return ANEURALNETWORKS_ERROR_DELETED.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be destroyed. Passing NULL is acceptable
 * and results in no operation.
 */
inline void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
  LOAD_FUNCTION(ANeuralNetworksExecution_free);
  EXECUTE_FUNCTION(execution);
}

/**
 * Associate a user buffer with an input of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided buffer must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This should be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other properties of the type must be the same as
 *             specified in the model. If the type is the same as specified
 *             when the model was built, NULL can be passed.
 * @param buffer The buffer containing the data.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the input.
 */
inline int ANeuralNetworksExecution_setInput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const void* buffer, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setInput);
  EXECUTE_FUNCTION_RETURN(execution, index, type, buffer, length);
}

/**
 * Associate part of a memory object with an input of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided memory must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the input argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param memory The memory containing the data.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The size in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the input.
 */
inline int ANeuralNetworksExecution_setInputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setInputFromMemory);
  EXECUTE_FUNCTION_RETURN(execution, index, type, memory, offset, length);
}

/**
 * Associate a user buffer with an output of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided buffer must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param buffer The buffer where the data is to be written.
 * @param length The length in bytes of the buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the output.
 */
inline int ANeuralNetworksExecution_setOutput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, void* buffer, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setOutput);
  EXECUTE_FUNCTION_RETURN(execution, index, type, buffer, length);
}

/**
 * Associate part of a memory object with an output of the model of the
 * {@link ANeuralNetworksExecution}.
 *
 * <p>The provided memory must outlive the execution.</p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be modified.
 * @param index The index of the output argument we are setting. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param type The type of the operand. This can be used to specify the
 *             dimensions that were set to 0 when the operand was added to the
 *             model. All other values must be the same as specified in the
 *             model. If the type is the same as specified when the model
 *             was built, NULL can be passed.
 * @param memory The memory where the data is to be stored.
 * @param offset This specifies the location of the data within the memory.
 *               The offset is in bytes from the start of memory.
 * @param length The length in bytes of the data value.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA if
 * the name is not recognized or the buffer is too small for the output.
 */
inline int ANeuralNetworksExecution_setOutputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    size_t offset, size_t length) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setOutputFromMemory);
  EXECUTE_FUNCTION_RETURN(execution, index, type, memory, offset, length);
}

/**
 * Schedule evaluation of the execution.
 *
 * <p>Schedules evaluation of the execution. Once the model has been
 * applied and the outputs are ready to be consumed, the execution will be
 * signaled. Use {@link ANeuralNetworksExecution_wait} to wait for that signal.
 * </p>
 *
 * Multiple executions can be scheduled and evaluated concurrently, and
 * compilations can be performed concurrently with executions. The runtime makes
 * no guarantee on the ordering of the completion of compilations and
 * executions. If it's important to the application, the application should
 * enforce the ordering by using {@link ANeuralNetworksCompilation_wait} and
 * {@link ANeuralNetworksExecution_wait}.
 *
 * ANeuralNetworksExecution_wait must be called to recuperate the resources used
 * by the execution.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @param execution The execution to be scheduled and executed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksExecution_startCompute(
    ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event) {
  LOAD_FUNCTION(ANeuralNetworksExecution_startCompute);
  EXECUTE_FUNCTION_RETURN(execution, event);
}

/**
 * Waits until the execution completes.
 *
 * More than one thread can wait on an event. When the execution completes,
 * all threads will be released.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
 */
inline int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
  LOAD_FUNCTION(ANeuralNetworksEvent_wait);
  EXECUTE_FUNCTION_RETURN(event);
}

/**
 * Destroys the event.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 */
inline void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
  LOAD_FUNCTION(ANeuralNetworksEvent_free);
  EXECUTE_FUNCTION(event);
}

/**
 * Get the number of available devices.
 *
 * @param numDevices Used to return the number of devices.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
inline int ANeuralNetworks_getDeviceCount(uint32_t* numDevices) {
  LOAD_FUNCTION(ANeuralNetworks_getDeviceCount);
  EXECUTE_FUNCTION_RETURN(numDevices);
}

/**
 * Get the representation of the specified device.
 *
 * @param devIndex The index of the specified device. Must be less than the
 *                 number of available devices.
 * @param device The representation of the specified device.
 *               The same representation will always be returned for the
 *               specified device.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */

inline int ANeuralNetworks_getDevice(uint32_t devIndex,
                                     ANeuralNetworksDevice** device) {
  LOAD_FUNCTION(ANeuralNetworks_getDevice);
  EXECUTE_FUNCTION_RETURN(devIndex, device);
}

/**
 * Get the name of the specified device.
 *
 * @param device The representation of the specified device.
 * @param name   The returned name of the specified device. The name will be in
 *               UTF-8 and will be null-terminated. It will be recognizable as a
 *               known device name rather than a cryptic string. For devices
 *               with API level 29 and above, the format of the name is
 *               {VENDOR}-{DEVICE}, e.g. “google-ipu”. For devices with feature
 *               level 28 or lower, the name will always be “unknown-device”.
 *               The name will remain valid for the duration of the application.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
inline int ANeuralNetworksDevice_getName(const ANeuralNetworksDevice* device,
                                         const char** name) {
  LOAD_FUNCTION(ANeuralNetworksDevice_getName);
  EXECUTE_FUNCTION_RETURN(device, name);
}

/**
 * Get the version of the driver implementation of the specified device.
 *
 * It’s the responsibility of the driver implementor to insure that this version
 * string uniquely distinguishes this implementation from all previous
 * implementations.
 *
 * This version string must not be confused with the feature level which is
 * solely defined by {@link ANeuralNetworksDevice_getFeatureLevel}. There is no
 * implicit ordering of the versions. For example, it is not possible to filter
 * all drivers older than a certain version.
 *
 * Application developers may use this version string to avoid or prefer
 * specific driver implementations. For example, an application may want to do
 * so because:
 *     - A specific version of the driver does not provide the required
 * performance, perhaps because of a performance regression.
 *     - A specific version of the driver has a bug or returns results that
 * don’t match the minimum precision requirement for the application.
 *
 * @param device  The representation of the specified device.
 * @param version The returned version string of the driver for the specified
 *                device. The string will be in UTF-8 and will be
 *                null-terminated. For devices with feature level 28 or lower,
 *                "UNKNOWN" will be returned. The version string will remain
 *                valid for the duration of the application.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
inline int ANeuralNetworksDevice_getVersion(const ANeuralNetworksDevice* device,
                                            const char** version) {
  LOAD_FUNCTION(ANeuralNetworksDevice_getVersion);
  EXECUTE_FUNCTION_RETURN(device, version);
}

/**
 * Get the supported NNAPI version of the specified device.
 *
 * Each device has a supported feature level, which is the most advanced feature
 * this driver implements. For example, if the driver implements the features
 * introduced in Android P, but does not implement the features introduced after
 * Android P, the value would be 28. Developers could decide whether or not the
 * specified device should be used for a Model that has certain feature
 * requirements.
 *
 * @param device       The representation of the specified device.
 * @param featureLevel The API level of the most advanced feature this driver
 *                     implements.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
inline int ANeuralNetworksDevice_getFeatureLevel(
    const ANeuralNetworksDevice* device, int64_t* featureLevel) {
  LOAD_FUNCTION(ANeuralNetworksDevice_getFeatureLevel);
  EXECUTE_FUNCTION_RETURN(device, featureLevel);
}

/**
 * Get the supported operations for a specified set of devices. If multiple
 * devices are selected, the supported operation list is a union of supported
 * operations of all selected devices.
 *
 * @param model        The model to be queried.
 * @param devices      The set of devices. Must not contain duplicates.
 * @param numDevices   The number of devices in the set.
 * @param supportedOps The boolean array to be filled. True means supported. The
 *                     size of the boolean array must be at least as large as
 *                     the number of operations in the model. The order of
 *                     elements in the supportedOps array matches the order in
 *                     which the corresponding operations were added to the
 *                     model.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
inline int ANeuralNetworksModel_getSupportedOperationsForDevices(
    const ANeuralNetworksModel* model,
    const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
    bool* supportedOps) {
  LOAD_FUNCTION(ANeuralNetworksModel_getSupportedOperationsForDevices);
  EXECUTE_FUNCTION_RETURN(model, devices, numDevices, supportedOps);
}

/**
 * Create a {@link ANeuralNetworksCompilation} to compile the given model for a
 * specified set of devices. If more than one device is specified, the
 * compilation will distribute the workload automatically across the devices.
 * The model must be fully supported by the specified set of devices. This means
 * that ANeuralNetworksModel_getSupportedOperationsForDevices() must have
 * returned true for every operation for that model/devices pair.
 *
 * @param model       The {@link ANeuralNetworksModel} to be compiled.
 * @param devices     The set of devices. Must not contain duplicates.
 * @param numDevices  The number of devices in the set.
 * @param compilation The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the model is invalid.
 *
 * Available since API level 29.
 */
inline int ANeuralNetworksCompilation_createForDevices(
    ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
    uint32_t numDevices, ANeuralNetworksCompilation** compilation) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_createForDevices);
  EXECUTE_FUNCTION_RETURN(model, devices, numDevices, compilation);
}

/**
 * Sets the compilation caching signature and the cache directory.
 *
 * Provides optional caching information to the runtime for faster repeated
 * compilation.
 *
 * See {@link ANeuralNetworksCompilation} for information on multithreaded
 * usage.
 *
 * @param compilation The compilation to be modified.
 * @param cacheDir The cache directory to store and retrieve caching data. It is
 *                 recommended to use the code_cache provided by the Android
 *                 runtime. If not using the code_cache, the user should choose
 *                 a directory local to the application, and is responsible to
 *                 manage and clean the cache entries.
 * @param token The token provided by the user to specify a model, must be of
 *              length ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN. The user should
 *              ensure that the token is unique to a model within the
 *              application. The NNAPI runtime will not detected token
 *              collisions. If there is a collision, the compilation outcome may
 *              be incorrect without notifying with error.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *
 * Available since API level 29.
 */
inline int ANeuralNetworksCompilation_setCaching(
    ANeuralNetworksCompilation* compilation, const char* cacheDir,
    const uint8_t* token) {
  LOAD_FUNCTION(ANeuralNetworksCompilation_setCaching);
  EXECUTE_FUNCTION_RETURN(compilation, cacheDir, token);
}

/**
 * Schedule synchronous evaluation of the execution.
 *
 * <p>Schedules synchronous evaluation of the execution. Returns once the
 * execution has completed and the outputs are ready to be consumed.
 * </p>
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * See {@link ANeuralNetworksExecution_startCompute} for asynchronous execution.
 * Synchronous execution incurs lower overhead than asynchronous execution.
 *
 * Available since API level 29.
 *
 * @param execution The execution to be scheduled and executed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
 *         ANEURALNETWORKS_UNMAPPABLE if the execution input or output memory
 *         cannot be properly mapped.
 */
inline int ANeuralNetworksExecution_compute(
    ANeuralNetworksExecution* execution) {
  LOAD_FUNCTION(ANeuralNetworksExecution_compute);
  EXECUTE_FUNCTION_RETURN(execution);
}

/**
 * Get the dimensional information of the specified output operand of the model
 * of the
 * {@link ANeuralNetworksExecution}.
 *
 * On asynchronous execution initiated by {@link
 * ANeuralNetworksExecution_startCompute},
 * {@link ANeuralNetworksEvent_wait} must be called prior to this function to
 * recuperate the resources used by the execution.
 *
 * @param execution The execution to be queried.
 * @param index The index of the output argument we are querying. It is
 *              an index into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with {@link
 * ANeuralNetworksModel_addOperand}.
 * @param rank The rank of the output operand.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful,
 * ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE if the target output is provided an
 * insufficient buffer at execution time, ANEURALNETWORKS_BAD_DATA if the index
 * is invalid.
 *
 * Available since API level 29.
 */
inline int ANeuralNetworksExecution_getOutputOperandRank(
    ANeuralNetworksExecution* execution, int32_t index, uint32_t* rank) {
  LOAD_FUNCTION(ANeuralNetworksExecution_getOutputOperandRank);
  EXECUTE_FUNCTION_RETURN(execution, index, rank);
}

/**
 * Get the dimensional information of the specified output operand of the model
 * of the
 * {@link ANeuralNetworksExecution}. The target output operand cannot be a
 * scalar.
 *
 * On asynchronous execution initiated by
 * {@link ANeuralNetworksExecution_startCompute},
 * {@link ANeuralNetworksEvent_wait} must be called prior to this function to
 * recuperate the resources used by the execution.
 *
 * @param execution The execution to be queried.
 * @param index The index of the output argument we are querying. It is an index
 *              into the lists passed to
 *              {@link ANeuralNetworksModel_identifyInputsAndOutputs}. It is not
 *              the index associated with
 *              {@link ANeuralNetworksModel_addOperand}.
 * @param dimensions The dimension array to be filled. The size of the array
 *                   must be exactly as large as the rank of the output operand
 *                   to be queried in the model.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful,
 * ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE if the target output is provided an
 * insufficient buffer at execution time, ANEURALNETWORKS_BAD_DATA if the index
 * is invalid or if the target is a scalar.
 *
 * Available since API level 29.
 */
inline int ANeuralNetworksExecution_getOutputOperandDimensions(
    ANeuralNetworksExecution* execution, int32_t index, uint32_t* dimensions) {
  LOAD_FUNCTION(ANeuralNetworksExecution_getOutputOperandDimensions);
  EXECUTE_FUNCTION_RETURN(execution, index, dimensions);
}

/**
 * Create a {@link ANeuralNetworksBurst} to apply the given compilation.
 * This only creates the burst object. Computation is only performed once
 * {@link ANeuralNetworksExecution_burstCompute} is invoked with a valid
 * {@link ANeuralNetworksExecution} and {@link ANeuralNetworksBurst}.
 *
 * <p>The provided compilation must outlive the burst object.</p>
 *
 * Available since API level 29.
 *
 * @param compilation The {@link ANeuralNetworksCompilation} to be evaluated.
 * @param burst The newly created object or NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful, ANEURALNETWORKS_BAD_DATA
 *         if the compilation is invalid.
 */
inline int ANeuralNetworksBurst_create(ANeuralNetworksCompilation* compilation,
                                       ANeuralNetworksBurst** burst) {
  LOAD_FUNCTION(ANeuralNetworksBurst_create);
  EXECUTE_FUNCTION_RETURN(compilation, burst);
}

/**
 * Destroys the burst object.
 *
 * Available since API level 29.
 *
 * @param burst The burst object to be destroyed. Passing NULL is acceptable and
 *              results in no operation.
 */
inline void ANeuralNetworksBurst_free(ANeuralNetworksBurst* burst) {
  LOAD_FUNCTION(ANeuralNetworksBurst_free);
  EXECUTE_FUNCTION(burst);
}

/**
 * Schedule synchronous evaluation of the execution on a burst object.
 *
 * <p>Schedules synchronous evaluation of the execution. Returns once the
 * execution has completed and the outputs are ready to be consumed.</p>
 *
 * <p>There must be at most one {@link ANeuralNetworksExecution} processing at
 * any given time for any given burst object. Any
 * {@link ANeuralNetworksExecution} launched before the previous has finished
 * will result in ANEURALNETWORKS_BAD_STATE.</p>
 *
 * Available since API level 29.
 *
 * @param burst The burst object to execute on.
 * @param execution The execution to be scheduled and executed. The execution
 *                  must be created from the same {@link
 *                  ANeuralNetworksCompilation} as the burst object.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the execution completed normally.
 */
inline int ANeuralNetworksExecution_burstCompute(
    ANeuralNetworksExecution* execution, ANeuralNetworksBurst* burst) {
  LOAD_FUNCTION(ANeuralNetworksExecution_burstCompute);
  EXECUTE_FUNCTION_RETURN(execution, burst);
}

/**
 * Creates a shared memory object from an AHardwareBuffer handle.
 *
 * If the shared memory is backed by an AHardwareBuffer of
 * AHARDWAREBUFFER_FORMAT_BLOB format, it can be used the same way as shared
 * memory created from a file handle. See
 * {@link ANeuralNetworksMemory} for a description on how to use this shared
 * memory.
 *
 * If the shared memory is backed by an AHardwareBuffer of a format other than
 * AHARDWAREBUFFER_FORMAT_BLOB, it can only be used for Model inputs and
 * outputs. When calling {@link ANeuralNetworksExecution_setInputFromMemory} or
 * {@link ANeuralNetworksExecution_setOutputFromMemory} with the shared memory,
 * both offset and length must be set to zero and the entire memory region will
 * be associated with the specified input or output operand. There is no
 * guarantee that an arbitrary AHardwareBuffer_Format and
 * AHardwareBuffer_UsageFlags combination can be used by arbitrary devices. The
 * execution will fail if selected set of devices cannot consume the buffer.
 *
 * Calling {@link ANeuralNetworksModel_setOperandValueFromMemory} with shared
 * memory backed by an AHardwareBuffer of a format other than
 * AHARDWAREBUFFER_FORMAT_BLOB is disallowed.
 *
 * TODO(miaowang): add documentation about intended usage with introspection
 * API.
 *
 * Available since API level 29.
 *
 * @param ahwb The AHardwareBuffer handle.
 * @param memory The memory object to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if the request completed normally.
 *
 * @see AHardwareBuffer
 */
inline int ANeuralNetworksMemory_createFromAHardwareBuffer(
    const AHardwareBuffer* ahwb, ANeuralNetworksMemory** memory) {
  LOAD_FUNCTION(ANeuralNetworksMemory_createFromAHardwareBuffer);
  EXECUTE_FUNCTION_RETURN(ahwb, memory);
}

/**
 * Specifies whether duration of the {@link ANeuralNetworksExecution} is to be
 * measured. By default, duration is not measured.
 *
 * The {@link ANeuralNetworksExecution} must have been created with
 * {@link ANeuralNetworksCompilation_createForDevices} with numDevices = 1.
 *
 * See {@link ANeuralNetworksExecution} for information on multithreaded usage.
 *
 * Available since API level 29.
 *
 * @param execution The execution to be modified.
 * @param measure 'true' if duration is to be measured, 'false' if not.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksExecution_setMeasureTiming(
    ANeuralNetworksExecution* execution, bool measure) {
  LOAD_FUNCTION(ANeuralNetworksExecution_setMeasureTiming);
  EXECUTE_FUNCTION_RETURN(execution, measure);
}

/**
 * Get the time spent in the specified {@link ANeuralNetworksExecution}, in
 * nanoseconds. The execution must have completed.
 *
 * @param execution The execution to be queried.
 * @param durationCode The measurement to be queried, specified by {@link
 * DurationCode}.
 * @param duration The returned duration. If no measurement was requested by
 *                 {@link ANeuralNetworksExecution_setMeasureTiming}, or for
 * some other reason the duration is not available, UINT64_MAX will be returned.
 *                 A particular device need not support any given measurement.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
inline int ANeuralNetworksExecution_getDuration(
    const ANeuralNetworksExecution* execution, int32_t durationCode,
    uint64_t* duration) {
  LOAD_FUNCTION(ANeuralNetworksExecution_getDuration);
  EXECUTE_FUNCTION_RETURN(execution, durationCode, duration);
}

/**/

#endif  // TENSORFLOW_LITE_NNAPI_NEURALNETWORKSSHIM_H_
