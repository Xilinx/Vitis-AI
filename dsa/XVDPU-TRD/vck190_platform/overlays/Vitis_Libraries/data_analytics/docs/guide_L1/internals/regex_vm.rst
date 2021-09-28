.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _guide-regex-VM:

*********************************************
Regular Expression Virtual Machine (regex-VM)
*********************************************


Overview
========

The regex-VM aims at the work of converting unstructured texts, like log files, into structured ones. Therefore, state-of-art high-throughput matching algorithm for comparing one string with many patterns, like that used in hyperscan, cannot work well in our target context. We chose VM-based approach, as it allows us to offer drop-in replacement in popular data transformation tools with regex often written in dialect of Perl, Python or Ruby.

The regex-VM consists of two parts: a software compiler written in C and a hardware virtual machine (VM) in C++.

1. Software compiler: compiles any regular expression given by user into an instruction list along with the corresponding bit-set map, number of instructions/character-classes/capturing-groups, and the name of each capturing group (if specified in input pattern).

2. Hardware VM: which takes the outputs from the compiler mentioned above to construct a practical matcher to match the string given in message buffer, and emit a 2-bit match flag indicating whether the input string is matched with the pattern or an internal stack overflow is happened. Futhermore, if the input string is matched, the offset addresses for each capturing group is provided in the output offset buffer, users can find the sub-strings in interest by picking them out from the whole input string according to the information given in that buffer.


User Guide
==========

Our regex-VM is implemented with VM approach, in which the pattern is translated into a series of instructions of a specialized VM, and the string being matched for drives the instruction jumps. We re-used VM instructions and compiler from popular `Oniguruma`_ library, which is the foundation of current Ruby `regex implementation`_, as our base.

.. _`Oniguruma`: https://github.com/kkos/oniguruma.git

.. _`regex implementation`: https://github.com/k-takata/Onigmo

Regex-VM Coverage
-----------------

Due to limited resources and a strict release timeline, in current release, we only provide the most frequently used OPs in the whole OP list given by Oniguruma library to support common regular expressions. This can be shown in the following table:

+-----------------------------+----------------+
| OP list                     | Supported      |
+-----------------------------+----------------+
| FINISH                      | NO             |
+-----------------------------+----------------+
| END                         | YES            |
+-----------------------------+----------------+
| STR_1                       | YES            |
+-----------------------------+----------------+
| STR_2                       | YES            |
+-----------------------------+----------------+
| STR_3                       | YES            |
+-----------------------------+----------------+
| STR_4                       | YES            |
+-----------------------------+----------------+
| STR_5                       | YES            |
+-----------------------------+----------------+
| STR_N                       | YES            |
+-----------------------------+----------------+
| STR_MB2N1                   | NO             |
+-----------------------------+----------------+
| STR_MB2N2                   | NO             |
+-----------------------------+----------------+
| STR_MB2N3                   | NO             |
+-----------------------------+----------------+
| STR_MB2N                    | NO             |
+-----------------------------+----------------+
| STR_MB3N                    | NO             |
+-----------------------------+----------------+
| STR_MBN                     | NO             |
+-----------------------------+----------------+
| CCLASS                      | YES            |
+-----------------------------+----------------+
| CCLASS_MB                   | NO             |
+-----------------------------+----------------+
| CCLASS_MIX                  | NO             |
+-----------------------------+----------------+
| CCLASS_NOT                  | YES            |
+-----------------------------+----------------+
| CCLASS_MB_NOT               | NO             |
+-----------------------------+----------------+
| CCLASS_MIX_NOT              | NO             |
+-----------------------------+----------------+
| ANYCHAR                     | YES            |
+-----------------------------+----------------+
| ANYCHAR_ML                  | NO             |
+-----------------------------+----------------+
| ANYCHAR_STAR                | YES            |
+-----------------------------+----------------+
| ANYCHAR_ML_STAR             | NO             |
+-----------------------------+----------------+
| ANYCHAR_STAR_PEEK_NEXT      | NO             |
+-----------------------------+----------------+
| ANYCHAR_ML_STAR_PEEK_NEXT   | NO             |
+-----------------------------+----------------+
| WORD                        | NO             |
+-----------------------------+----------------+
| WORD_ASCII                  | NO             |
+-----------------------------+----------------+
| NO_WROD                     | NO             |
+-----------------------------+----------------+
| NO_WORD_ASCII               | NO             |
+-----------------------------+----------------+
| WORD_BOUNDARY               | NO             |
+-----------------------------+----------------+
| NO_WORD_BOUNDARY            | NO             |
+-----------------------------+----------------+
| WORD_BEGIN                  | NO             |
+-----------------------------+----------------+
| WORD_END                    | NO             |
+-----------------------------+----------------+
| TEXT_SEGMENT_BOUNDARY       | NO             |
+-----------------------------+----------------+
| BEGIN_BUF                   | YES            |
+-----------------------------+----------------+
| END_BUF                     | YES            |
+-----------------------------+----------------+
| BEGIN_LINE                  | YES            |
+-----------------------------+----------------+
| END_LINE                    | YES            |
+-----------------------------+----------------+
| SEMI_END_BUF                | NO             |
+-----------------------------+----------------+
| CHECK_POSITION              | NO             |
+-----------------------------+----------------+
| BACKREF1                    | NO             |
+-----------------------------+----------------+
| BACKREF2                    | NO             |
+-----------------------------+----------------+
| BACKREF_N                   | NO             |
+-----------------------------+----------------+
| BACKREF_N_IC                | NO             |
+-----------------------------+----------------+
| BACKREF_MULTI               | NO             |
+-----------------------------+----------------+
| BACKREF_MULTI_IC            | NO             |
+-----------------------------+----------------+
| BACKREF_WITH_LEVEL          | NO             |
+-----------------------------+----------------+
| BACKREF_WITH_LEVEL_IC       | NO             |
+-----------------------------+----------------+
| BACKREF_CHECK               | NO             |
+-----------------------------+----------------+
| BACKREF_CHECK_WITH_LEVEL    | NO             |
+-----------------------------+----------------+
| MEM_START                   | YES            |
+-----------------------------+----------------+
| MEM_START_PUSH              | YES            |
+-----------------------------+----------------+
| MEM_END_PUSH                | NO             |
+-----------------------------+----------------+
| MEM_END_PUSH_REC            | NO             |
+-----------------------------+----------------+
| MEM_END                     | YES            |
+-----------------------------+----------------+
| MEM_END_REC                 | NO             |
+-----------------------------+----------------+
| FAIL                        | YES            |
+-----------------------------+----------------+
| JUMP                        | YES            |
+-----------------------------+----------------+
| PUSH                        | YES            |
+-----------------------------+----------------+
| PUSH_SUPER                  | NO             |
+-----------------------------+----------------+
| POP                         | YES            |
+-----------------------------+----------------+
| POP_TO_MARK                 | YES            |
+-----------------------------+----------------+
| PUSH_OR_JUMP_EXACT1         | YES            |
+-----------------------------+----------------+
| PUSH_IF_PEEK_NEXT           | NO             |
+-----------------------------+----------------+
| REPEAT                      | YES            |
+-----------------------------+----------------+
| REPEAT_NG                   | NO             |
+-----------------------------+----------------+
| REPEAT_INC                  | YES            |
+-----------------------------+----------------+
| REPEAT_INC_NG               | NO             |
+-----------------------------+----------------+
| EMPTY_CHECK_START           | NO             |
+-----------------------------+----------------+
| EMPTY_CHECK_END             | NO             |
+-----------------------------+----------------+
| EMPTY_CHECK_END_MEMST       | NO             |
+-----------------------------+----------------+
| EMPTY_CHECK_END_MEMST_PUSH  | NO             |
+-----------------------------+----------------+
| MOVE                        | NO             |
+-----------------------------+----------------+
| STEP_BACK_START             | YES            |
+-----------------------------+----------------+
| STEP_BACK_NEXT              | NO             |
+-----------------------------+----------------+
| CUT_TO_MARK                 | NO             |
+-----------------------------+----------------+
| MARK                        | YES            |
+-----------------------------+----------------+
| SAVE_VAL                    | NO             |
+-----------------------------+----------------+
| UPDATE_VAR                  | NO             |
+-----------------------------+----------------+
| CALL                        | NO             |
+-----------------------------+----------------+
| RETURN                      | NO             |
+-----------------------------+----------------+
| CALLOUT_CONTECTS            | NO             |
+-----------------------------+----------------+
| CALLOUT_NAME                | NO             |
+-----------------------------+----------------+

Therefore, the supported atomic regular expressions and their corresponding descriptions should be:

+-------------------+------------------------------------------------------------------------------------------------------+
| Regex             | Description                                                                                          |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``^``             | asserts position at start of a line                                                                  |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``$``             | asserts position at the end of a line                                                                |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\A``            | asserts position at start of the string                                                              |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\z``            | asserts position at the end of the string                                                            |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\ca``           | matches the control sequence ``CTRL+a``                                                              |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\C``            | matches one data unit, even in UTF mode (best avoided)                                               |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\c\\``          | matches the control sequence ``CTRL+\``                                                              |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\s``            | matches any whitespace character (equal to ``[\r\n\t\f\v ]``)                                        |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\S``            | matches any non-whitespace character (equal to ``[^\r\n\t\f\v ]``)                                   |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\d``            | matches a digit (equal to ``[0-9]``)                                                                 |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\D``            | matches any character that's not a digit (equal to ``[^0-9]``)                                       |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\h``            | matches any horizontal whitespace character (equal to ``[[:blank:]]``)                               |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\H``            | matches any character that's not a horizontal whitespace character                                   |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\w``            | matches any word character (equal to ``[a-zA-Z0-9_]``)                                               |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\W``            | matches any non-word character (equal to ``[^a-zA-Z0-9_]``)                                          |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\^``            | matches the character ``^`` literally                                                                |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\$``            | matches the character ``$`` literally                                                                |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\N``            | matches any non-newline character                                                                    |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\g'0'``         | recurses the 0th subpattern                                                                          |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\o{101}``       | matches the character ``A`` with index with ``101(oct)``                                             |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\x61``          | matches the character ``a (hex 61)`` literally                                                       |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\x{1 2}``       | matches ``1 (hex)`` or ``2 (hex)``                                                                   |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``\17``           | matches the character ``oct 17`` literally                                                           |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``abc``           | matches the ``abc`` literally                                                                        |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``.``             | matches any character (except for line terminators)                                                  |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``|``             | alternative                                                                                          |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``[^a]``          | match a single character not present in the list below                                               |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``[a-c]``         | matches ``a``, ``b``, or ``c``                                                                       |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``[abc]``         | matches ``a``, ``b``, or ``c``                                                                       |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``[:upper:]``     | matches a uppercase letter ``[A-Z]``                                                                 |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``a?``            | matches the ``a`` zero or one time (**greedy**)                                                      |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``a*``            | matches ``a`` between zero and unlimited times (**greedy**)                                          |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``a+``            | matches ``a`` between one and unlimited times (**greedy**)                                           |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``a??``           | matches ``a`` between zero and one times (**lazy**)                                                  |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``a*?``           | matches ``a`` between zero and unlimited times (**lazy**)                                            |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``a+?``           | matches ``a`` between one and unlimited times (**lazy**)                                             |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``a{2}``          | matches ``a`` exactly 2 times                                                                        |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``a{0,}``         | matches ``a`` between zero and unlimited times                                                       |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``a{1,2}``        | matches ``a`` one or two times                                                                       |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``{,}``           | matches ``{,}`` literally                                                                            |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``(?#blabla)``    | comment ``blabla``                                                                                   |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``(a)``           | capturing group, matches ``a`` literally                                                             |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``(?<name1> a)``  | named capturing group ``name1``, matches ``a`` literally                                             |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``(?:)``          | non-capturing group                                                                                  |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``(?i)``          | match the remainder of the pattern with the following effective flags: gmi (i modifier: insensitive) |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``(?<!a)z``       | matches any occurrence of ``z`` that is not preceded by ``a`` (negative look-behind)                 |
+-------------------+------------------------------------------------------------------------------------------------------+
| ``z(?!a)``        | match any occurrence of ``z`` that is not followed by ``a`` (negative look-ahead)                    |
+-------------------+------------------------------------------------------------------------------------------------------+

.. ATTENTION::
    1. Supported encoding method in current release is ASCII (extended ASCII codes are excluded).
    2. Nested repetition is not supported

Regex-VM Usage
--------------

Before instantiating the hardware VM, users have to pre-compile their regular expression using the software compiler mentioned above first to check if the pattern is supported by the hardware VM. The compiler will give an error code ``XF_UNSUPPORTED_OPCODE`` if the pattern is not supported. A pass code ``ONIG_NORMAL`` along with the configurations (including instruction list, bit-set map etc.) will be given if the input is a valid pattern. Then, user should pass these configurations and the input message with its corresponding length in bytes to the hardware VM to trigger the matching process. The hardware VM will judge whether the input message is matched and provide the offset addresses for each capturing group in offset buffer.

It is important to be noticed that only the internal stack buffer is hold in hardware VM, user should allocate memories for bit-set map, instruction buffer, message buffer accordingly, and offset buffer respectively outside the hardware instantiation.

For the internal stack, its size is decided by the template parameter of the hardware VM. Since the storage resource it uses is URAM, the ``STACK_SIZE`` should better be set to be a multiple of 4096 for not wasting the space of individual URAM block. Moreover, it is critical to choose the internal stack size wisely as the hardware VM will overflow if the size is too small or no URAMs will be available on board for you to instantiate more PUs to improve the throughput.

**Code Example**

The following section gives a usage example for using regex-VM in C++ based HLS design.

To use the regex-VM you need to:

1. Compile the software regular expression compiler by running ``make`` command in path ``L1/tests/text/regex_vm/re_compile``

2. Include the ``xf_re_compile.h`` header in path ``L1/include/sw/xf_data_analytics/text`` and the ``oniguruma.h`` header in path ``L1/tests/text/regex_vm/re_compile/lib/include``

.. code-block:: cpp

    #include "oniguruma.h"
    #include "xf_re_compile.h"

3. Compile your regular expression by calling ``xf_re_compile``

.. code-block:: cpp

    int r = xf_re_compile(pattern, bitset, instr_buff, instr_num, cclass_num, cpgp_num, NULL, NULL);

4. Check the return value to see if its a valid pattern and supported by hardware VM. ``ONIG_NORMAL`` is returned if the pattern is valid, and ``XF_UNSUPPORTED_OPCODE`` is returned if it's not supported currently.

.. code-block:: cpp

    if (r != XF_UNSUPPORTED_OPCODE && r == ONIG_NORMAL) {
        // calling hardware VM here for acceleration
    }

5. Once the regular expression is verified as a supported pattern, you may call hardware VM to match any message you want by

.. code-block:: cpp

    // for data types used in VM
    #include "ap_int.h"
    // header for hardware VM implementation
    #include "xf_data_analytics/text/regexVM.hpp"

    // allocate memory for bit-set map
    unsigned int bitset[8 * cclass_num];
    // allocate memory for instruction buffer (derived from software compiler)
    uint64_t instr_buff[instr_num];
    // allocate memory for message
    ap_uint<32> msg_buff[MESSAGE_SIZE];
    // set up input message buffer according to input string
    unsigned str_len = strlen((const char*)in_str);
    for (int i = 0; i < (str_len + 3) / 4;  i++) {
        for (int k = 0; k < 4; k++) {
            if (i * 4 + k < str_len) {
                msg_buff[i].range((k + 1) * 8 - 1, k * 8) = in_str[i * 4 + k];
            } else {
                // pad white-space at the end
                msg_buff[i].range((k + 1) * 8 - 1, k * 8) = ' ';
            }
        }
    }
    // allocate memory for offset addresses for each capturing group
    uint16_t offset_buff[2 * (cpgp_num + 1)];
    // initialize offset buffer
    for (int i = 0; i < 2 * CAP_GRP_NUM; i++) {
        offset_buff[i] = -1;
    }
    ap_uint<2> match = 0;
    // call for hardware acceleration (basic hardware VM implementation)
    xf::data_analytics::text:regexVM<STACK_SIZE>((ap_uint<32>*)bitset, (ap_uint<64>*)instr_buff, msg_buff, str_len, match, offset_buff);
    // or call for hardware acceleration (performance optimized hardware VM implementation)
    xf::data_analytics::text:regexVM_opt<STACK_SIZE>((ap_uint<32>*)bitset, (ap_uint<64>*)instr_buff, msg_buff, str_len, match, offset_buff);

The match flag and offset addresses for each capturing group are presented in ``match`` and ``offset_buff`` respectively with the format shown in the tables below.

Truth table for the 2-bit output ``match`` flag of hardware VM:

+-------+-------------------------+
| Value | Description             |
+-------+-------------------------+
| 0     | mismatched              |
+-------+-------------------------+
| 1     | matched                 |
+-------+-------------------------+
| 2     | internal stack overflow |
+-------+-------------------------+
| 3     | reserved for future use |
+-------+-------------------------+

Arrangement of the offset buffer ``offsetBuff``:

+---------+---------------------------------------------+
| Address | Description                                 |
+---------+---------------------------------------------+
| 0       | start position of the whole matched string  |
+---------+---------------------------------------------+
| 1       | end position of the whole matched string    |
+---------+---------------------------------------------+
| 2       | start position of the 1st capturing group   |
+---------+---------------------------------------------+
| 3       | end position of the 1st capturing group     |
+---------+---------------------------------------------+
| 4       | start position of the 2nd capturing group   |
+---------+---------------------------------------------+
| 5       | end position of the 2nd capturing group     |
+---------+---------------------------------------------+
| ...     | ...                                         |
+---------+---------------------------------------------+


Implemention
============

If you go into the details of the implementation of hardware VM, you may find even the basic version of hardware VM is significantly different from the one in Oniguruma, let alone the performance optimized one. Thus, this section is especially for developers who wants to add more OPs to the VM by themselves or who are extremely interested in our design.

The first thing you want to conquer will be the software compiler. Once you have a full understanding of a specific OP in Oniguruma, you have to add it to the corresponding instruction with the format acceptable for the hardware VM. The 64-bit instruction format for communication between software compiler and hardware VM can be explained like this:

.. image:: /images/instruction_format.png
   :alt: Instruction Format
   :width: 80%
   :align: center

Then, if the OP you want to add is related to a jump/push operation on the OP address, the absolute address must be provided at the first while-loop in the source code of the software compiler for calculation of the address which will be put into instruction list later. The rest information related to this OP and the calculated address should be pack into one instruction at the second while-loop. So far, the software compiler part is done.

Location of the source of the software compiler: ``L1/src/sw/xf_re_compile.c``

Finally, add the corresponding logic to the hardware VM based on your understanding of the OP and test it accordingly. Once the test passed, you may start optimizing the implemtation which is extremely challenging and tricky.

Let me introduce you what we've done currently for optimizing the hardware VM. Hope it will inspire you to some extent.

1. Simplify the internal logic for each OP we added as mush as we can.

2. Merge the newly added OP into another if possible to let them share the same logic.

3. Offload runtime calculations to software compiler for pre-calculation if possible to improve the runtime performance.

4. Separate the data flow and control flow, do pre-fetch and post-store operations to improve memory access efficiency.

5. Resolve the read-and-write dependency of on-chip RAMs by caching intermediate data in registers to avoid unnecessary accesses.

6. Execute a predict (2nd) instruction in each iteration to accelerate the process under specific circumstances. (performance optimized version executes 2 instructions / 3 cycles)

.. NOTE::
    For the following scenarios, the predict instruction will not be executed:

    1. Read/write the internal stack simultaneously

    2. OP for 2nd instruction is any_char_star, pop_to_mark, or mem_start_push

    3. Jump on OP address happened in 1st instruction

    4. Read/write the offset buffer simultaneously

    5. Pointer for input string moves in 1st instruction and 2nd instruction goes into the OP which needs character comparision

    6. Write the offset buffer simultaneously


Profiling
=========

The hardware resource utilization of hardware VM is shown in the table below (performance optimized version at FMax = 352MHz).

+----------------+-------+------+--------+--------+------+-----+-----+
| Primitive      | CLB   |  LUT |   FF   |  BRAM  | URAM | DSP | SRL |
+----------------+-------+------+--------+--------+------+-----+-----+
| hardware VM    | 305   | 1690 |  973   |    0   |  4   |  0  | 0   |
+----------------+-------+------+--------+--------+------+-----+-----+

