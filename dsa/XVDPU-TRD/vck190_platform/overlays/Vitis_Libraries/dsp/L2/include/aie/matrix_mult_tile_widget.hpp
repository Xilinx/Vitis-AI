/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DSPLIB_CONDITIONAL_WIDGET_HPP_
#define _DSPLIB_CONDITIONAL_WIDGET_HPP_

// This file holds the definition of the conditional widget class
/**
 * @file conditional_widget.hpp
 *
 **/

#include <adf.h>
#include <vector>

namespace xf {
namespace dsp {
namespace aie {
namespace blas {
namespace matrix_mult {

using namespace adf;

template <unsigned int addWidget, unsigned int windowSize, class widgetClass>
class ConditionalWidget {
   public:
    using portConnect = connect<window<windowSize> >;
    ConditionalWidget(){}; // default constructor
    template <typename inType, typename outType>
    static kernel create(port<inType>& inPort, port<outType>& outPort) {
        kernel widget;
        if (addWidget == 1) {
            widget = kernel::create_object<widgetClass>();
            portConnect(inPort, widget.in[0]);
            portConnect(widget.out[0], outPort);
        } else {
            portConnect(inPort, outPort);
        }

        return widget;
    }
};
}
}
}
}
}

#endif