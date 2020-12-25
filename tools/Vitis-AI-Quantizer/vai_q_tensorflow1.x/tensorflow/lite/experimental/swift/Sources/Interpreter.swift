// Copyright 2018 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import TensorFlowLiteC

/// A TensorFlow Lite interpreter that performs inference from a given model.
public final class Interpreter {
  /// `TfLiteInterpreter` C pointer type represented as an `UnsafePointer<TFL_Interpreter>`.
  private typealias CInterpreter = OpaquePointer

  /// Total number of input tensors associated with the model.
  public var inputTensorCount: Int {
    return Int(TfLiteInterpreterGetInputTensorCount(cInterpreter))
  }

  /// Total number of output tensors associated with the model.
  public var outputTensorCount: Int {
    return Int(TfLiteInterpreterGetOutputTensorCount(cInterpreter))
  }

  /// Underlying `TfLiteInterpreter` C pointer.
  private var cInterpreter: CInterpreter?

  /// Creates a new model interpreter instance.
  ///
  /// - Parameters:
  ///   - modelPath: Local file path to a TensorFlow Lite model.
  ///   - options: Custom configurations for the interpreter. Default is `nil` indicating that the
  ///       interpreter will determine the configuration options.
  /// - Throws: An error if the model could not be loaded or the interpreter could not be created.
  public init(modelPath: String, options: InterpreterOptions? = nil) throws {
    guard let model = Model(filePath: modelPath) else { throw InterpreterError.failedToLoadModel }

    let cInterpreterOptions: OpaquePointer? = try options.map { options in
      guard let cOptions = TfLiteInterpreterOptionsCreate() else {
        throw InterpreterError.failedToCreateInterpreter
      }
      if let threadCount = options.threadCount, threadCount > 0 {
        TfLiteInterpreterOptionsSetNumThreads(cOptions, Int32(threadCount))
      }
      TfLiteInterpreterOptionsSetErrorReporter(
        cOptions,
        { (_, format, args) -> Void in
          // Workaround for optionality differences for x86_64 (non-optional) and arm64 (optional).
          let optionalArgs: CVaListPointer? = args
          guard let cFormat = format,
            let arguments = optionalArgs,
            let message = String(cFormat: cFormat, arguments: arguments)
          else {
            return
          }
          print(String(describing: InterpreterError.tensorFlowLiteError(message)))
        },
        nil
      )
      return cOptions
    }
    defer { TfLiteInterpreterOptionsDelete(cInterpreterOptions) }

    guard let cInterpreter = TfLiteInterpreterCreate(model.cModel, cInterpreterOptions) else {
      throw InterpreterError.failedToCreateInterpreter
    }
    self.cInterpreter = cInterpreter
  }

  deinit {
    TfLiteInterpreterDelete(cInterpreter)
  }

  /// Invokes the interpreter to perform inference from the loaded graph.
  ///
  /// - Throws: An error if the model was not ready because tensors were not allocated.
  public func invoke() throws {
    guard TfLiteInterpreterInvoke(cInterpreter) == kTfLiteOk else {
      throw InterpreterError.allocateTensorsRequired
    }
  }

  /// Returns the input tensor at the given index.
  ///
  /// - Parameters:
  ///   - index: The index for the input tensor.
  /// - Throws: An error if the index is invalid or the tensors have not been allocated.
  /// - Returns: The input tensor at the given index.
  public func input(at index: Int) throws -> Tensor {
    let maxIndex = inputTensorCount - 1
    guard case 0...maxIndex = index else {
      throw InterpreterError.invalidTensorIndex(index: index, maxIndex: maxIndex)
    }
    guard let cTensor = TfLiteInterpreterGetInputTensor(cInterpreter, Int32(index)),
      let bytes = TfLiteTensorData(cTensor),
      let nameCString = TfLiteTensorName(cTensor)
    else {
      throw InterpreterError.allocateTensorsRequired
    }
    guard let dataType = TensorDataType(type: TfLiteTensorType(cTensor)) else {
      throw InterpreterError.invalidTensorDataType
    }

    let name = String(cString: nameCString)
    let rank = TfLiteTensorNumDims(cTensor)
    let dimensions = (0..<rank).map { Int(TfLiteTensorDim(cTensor, $0)) }
    let shape = TensorShape(dimensions)
    let byteCount = TfLiteTensorByteSize(cTensor)
    let data = Data(bytes: bytes, count: byteCount)
    let cQuantizationParams = TfLiteTensorQuantizationParams(cTensor)
    let scale = cQuantizationParams.scale
    let zeroPoint = Int(cQuantizationParams.zero_point)
    var quantizationParameters: QuantizationParameters? = nil
    if scale != 0.0 {
      quantizationParameters = QuantizationParameters(scale: scale, zeroPoint: zeroPoint)
    }
    let tensor = Tensor(
      name: name,
      dataType: dataType,
      shape: shape,
      data: data,
      quantizationParameters: quantizationParameters
    )
    return tensor
  }

  /// Returns the output tensor at the given index.
  ///
  /// - Parameters:
  ///   - index: The index for the output tensor.
  /// - Throws: An error if the index is invalid, tensors haven't been allocated, or interpreter
  ///     hasn't been invoked for models that dynamically compute output tensors based on the values
  ///     of its input tensors.
  /// - Returns: The output tensor at the given index.
  public func output(at index: Int) throws -> Tensor {
    let maxIndex = outputTensorCount - 1
    guard case 0...maxIndex = index else {
      throw InterpreterError.invalidTensorIndex(index: index, maxIndex: maxIndex)
    }
    guard let cTensor = TfLiteInterpreterGetOutputTensor(cInterpreter, Int32(index)),
      let bytes = TfLiteTensorData(cTensor),
      let nameCString = TfLiteTensorName(cTensor)
    else {
      throw InterpreterError.invokeInterpreterRequired
    }
    guard let dataType = TensorDataType(type: TfLiteTensorType(cTensor)) else {
      throw InterpreterError.invalidTensorDataType
    }

    let name = String(cString: nameCString)
    let rank = TfLiteTensorNumDims(cTensor)
    let dimensions = (0..<rank).map { Int(TfLiteTensorDim(cTensor, $0)) }
    let shape = TensorShape(dimensions)
    let byteCount = TfLiteTensorByteSize(cTensor)
    let data = Data(bytes: bytes, count: byteCount)
    let cQuantizationParams = TfLiteTensorQuantizationParams(cTensor)
    let scale = cQuantizationParams.scale
    let zeroPoint = Int(cQuantizationParams.zero_point)
    var quantizationParameters: QuantizationParameters? = nil
    if scale != 0.0 {
      quantizationParameters = QuantizationParameters(scale: scale, zeroPoint: zeroPoint)
    }
    let tensor = Tensor(
      name: name,
      dataType: dataType,
      shape: shape,
      data: data,
      quantizationParameters: quantizationParameters
    )
    return tensor
  }

  /// Resizes the input tensor at the given index to the specified tensor shape.
  ///
  /// - Note: After resizing an input tensor, the client **must** explicitly call
  ///     `allocateTensors()` before attempting to access the resized tensor data or invoking the
  ///     interpreter to perform inference.
  /// - Parameters:
  ///   - index: The index for the input tensor.
  ///   - shape: The shape that the input tensor should be resized to.
  /// - Throws: An error if the input tensor at the given index could not be resized.
  public func resizeInput(at index: Int, to shape: TensorShape) throws {
    let maxIndex = inputTensorCount - 1
    guard case 0...maxIndex = index else {
      throw InterpreterError.invalidTensorIndex(index: index, maxIndex: maxIndex)
    }
    guard TfLiteInterpreterResizeInputTensor(
      cInterpreter,
      Int32(index),
      shape.int32Dimensions,
      Int32(shape.rank)
    ) == kTfLiteOk
    else {
      throw InterpreterError.failedToResizeInputTensor(index: index)
    }
  }

  /// Copies the given data to the input tensor at the given index.
  ///
  /// - Parameters:
  ///   - data: The data to be copied to the input tensor's data buffer.
  ///   - index: The index for the input tensor.
  /// - Throws: An error if the `data.count` does not match the input tensor's `data.count` or if
  ///     the given index is invalid.
  /// - Returns: The input tensor with the copied data.
  @discardableResult
  public func copy(_ data: Data, toInputAt index: Int) throws -> Tensor {
    let maxIndex = inputTensorCount - 1
    guard case 0...maxIndex = index else {
      throw InterpreterError.invalidTensorIndex(index: index, maxIndex: maxIndex)
    }
    guard let cTensor = TfLiteInterpreterGetInputTensor(cInterpreter, Int32(index)) else {
      throw InterpreterError.allocateTensorsRequired
    }

    let byteCount = TfLiteTensorByteSize(cTensor)
    guard data.count == byteCount else {
      throw InterpreterError.invalidTensorDataCount(provided: data.count, required: byteCount)
    }

    #if swift(>=5.0)
    let status = data.withUnsafeBytes {
      TfLiteTensorCopyFromBuffer(cTensor, $0.baseAddress, data.count)
    }
    #else
    let status = data.withUnsafeBytes { TfLiteTensorCopyFromBuffer(cTensor, $0, data.count) }
    #endif  // swift(>=5.0)
    guard status == kTfLiteOk else { throw InterpreterError.failedToCopyDataToInputTensor }
    return try input(at: index)
  }

  /// Allocates memory for all input tensors based on their `TensorShape`s.
  ///
  /// - Note: This is a relatively expensive operation and should only be called after creating the
  ///     interpreter and/or resizing any input tensors.
  /// - Throws: An error if memory could not be allocated for the input tensors.
  public func allocateTensors() throws {
    guard TfLiteInterpreterAllocateTensors(cInterpreter) == kTfLiteOk else {
      throw InterpreterError.failedToAllocateTensors
    }
  }
}

// MARK: - Extensions

extension String {
  /// Returns a new `String` initialized by using the given format C array as a template into which
  /// the remaining argument values are substituted according to the user’s default locale.
  ///
  /// - Note: Returns `nil` if a new `String` could not be constructed from the given values.
  /// - Parameters:
  ///   - cFormat: The format C array as a template for substituting values.
  ///   - arguments: A C pointer to a `va_list` of arguments to substitute into `cFormat`.
  init?(cFormat: UnsafePointer<CChar>, arguments: CVaListPointer) {
    var buffer: UnsafeMutablePointer<CChar>?
    guard vasprintf(&buffer, cFormat, arguments) != 0, let cString = buffer else { return nil }
    self.init(validatingUTF8: cString)
  }
}
