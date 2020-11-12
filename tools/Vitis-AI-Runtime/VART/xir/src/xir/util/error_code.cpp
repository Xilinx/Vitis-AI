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

#include "UniLog/ErrorCode.hpp"

// Common
REGISTER_ERROR_CODE(XIR_OUT_OF_RANGE, "idx out of range!", "");
REGISTER_ERROR_CODE(XIR_XIR_UNDEFINED_OPERATION, "Undefined operation!", "");
REGISTER_ERROR_CODE(XIR_UNEXPECTED_VALUE, "Unexpected value!", "");
REGISTER_ERROR_CODE(XIR_VALUE_UNMATCH, "Value unmatch!", "");

// Op def related
REGISTER_ERROR_CODE(XIR_UNREGISTERED_OP, "Unregistered operator!", "");
REGISTER_ERROR_CODE(XIR_MULTI_REGISTERED_OP,
                    "Multiple registration of operator!", "");
REGISTER_ERROR_CODE(XIR_UNREGISTERED_ATTR, "Unregistered attribute!", "");
REGISTER_ERROR_CODE(XIR_MULTI_REGISTERED_ATTR,
                    "Multiple registration of attribute!", "");
REGISTER_ERROR_CODE(XIR_UNREGISTERED_ARG, "Unregistered argument!", "");
REGISTER_ERROR_CODE(XIR_MULTI_REGISTERED_ARG,
                    "Multiple registration of argument!", "");
REGISTER_ERROR_CODE(XIR_OP_DEF_SHAPE_HINT_MISSING,
                    "A shape hint is required by the op definition", "");

// Op arg related
REGISTER_ERROR_CODE(XIR_UNDEFINED_INPUT_ARG, "Undefined input arg!", "");
REGISTER_ERROR_CODE(XIR_INVALID_ARG_OCCUR, "Invalid arg occurence!", "");

// Op attr related
REGISTER_ERROR_CODE(XIR_INVALID_ATTR_DEF, "Invalid attribute defination!", "");
REGISTER_ERROR_CODE(XIR_UNDEFINED_ATTR, "Undefined attribute!", "");
REGISTER_ERROR_CODE(XIR_INVALID_ATTR_OCCUR, "Invalid attr occurence!", "");

// Graph related
REGISTER_ERROR_CODE(XIR_UNDEFINED_OP, "Access undefined OP!", "");
REGISTER_ERROR_CODE(XIR_MULTI_DEFINED_OP, "Multiple definition of OP!", "");
REGISTER_ERROR_CODE(XIR_MULTI_DEFINED_TENSOR, "Multiple definition of Tensor!",
                    "");
REGISTER_ERROR_CODE(XIR_REMOVE_OP_FAIL, "Failed to remove an op!", "");
REGISTER_ERROR_CODE(XIR_ADD_OP_FAIL, "Failed to add an op!", "");
REGISTER_ERROR_CODE(XIR_OP_NAME_CONFLICT,
                    "There're at least two ops assigned the same name, please "
                    "check the ops' name.",
                    "");

// System operation related
REGISTER_ERROR_CODE(XIR_OPERATION_FAILED, "Fail to execute command!", "");
REGISTER_ERROR_CODE(XIR_FILE_NOT_EXIST, "File does't exist!", "");

// Subgraph related
REGISTER_ERROR_CODE(XIR_SUBGRAPH_CREATE_CHILDREN_FOR_NONLEAF, "", "");
REGISTER_ERROR_CODE(XIR_SUBGRAPH_INVALID_MERGE_REQUEST_NONCHILD,
                    "Cannot merge subgraphs which are not children!", "");
REGISTER_ERROR_CODE(XIR_SUBGRAPH_INVALID_MERGE_REQUEST_NONLEAF,
                    "Cannot merge subgraphs which are not leaves!", "");
REGISTER_ERROR_CODE(XIR_SUBGRAPH_ALREADY_CREATED_ROOT,
                    "Already created root subgraph for the graph!", "");
REGISTER_ERROR_CODE(XIR_SUBGRAPH_HAS_CYCLE,
                    "Children from a same subgraph depend each other!", "");

// Static Attr Related
REGISTER_ERROR_CODE(XIR_MULTI_REGISTERED_EXPANDED_ATTR,
                    "Multiple registration of static attr!", "");

// Tensor related
REGISTER_ERROR_CODE(XIR_MEANINGLESS_VALUE,
                    "The value you set for this parameter makes no sense.", "");
REGISTER_ERROR_CODE(XIR_ACCESS_ADDRESS_OVERFLOW,
                    "The address you try to access does not exsit!", "");
REGISTER_ERROR_CODE(
    XIR_PROTECTED_MEMORY,
    "The content in protected momory for tensor can not be modified!", "");
REGISTER_ERROR_CODE(XIR_UNKNOWNTYPE_TENSOR,
                    "The DataType of this tensor is not specified.", "");
REGISTER_ERROR_CODE(XIR_UNEQUIVALENT_ATTRIBUTE,
                    "These two attibutes/parameters are not equivalent.", "");
REGISTER_ERROR_CODE(XIR_INVALID_DATA_TYPE, "The data type is invalid.", "");
// pb releated
REGISTER_ERROR_CODE(XIR_UNSUPPORTED_TYPE,
                    "unsupported data type for attr value", "");
REGISTER_ERROR_CODE(XIR_READ_PB_FAILURE, "failed to read pb file", "");
REGISTER_ERROR_CODE(XIR_WRITE_PB_FAILURE, "failed to write pb file", "");
REGISTER_ERROR_CODE(XIR_SHAPE_UNMATCH, "The shape is unmatching.", "");
// internal error
REGISTER_ERROR_CODE(XIR_INTERNAL_ERROR,
                    "it is an internal bug supposed never happen", "");
REGISTER_ERROR_CODE(XIR_UNSUPPORTED_ROUND_MODE, "unsupported round mode.", "");
