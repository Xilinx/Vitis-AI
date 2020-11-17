/**
 * Copyright (C) 2018 - 2019 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef __SectionEmbeddedMetadata_h_
#define __SectionEmbeddedMetadata_h_

// ----------------------- I N C L U D E S -----------------------------------

// #includes here - please keep these to a bare minimum!
#include "Section.h"
#include <boost/functional/factory.hpp>

// ------------ F O R W A R D - D E C L A R A T I O N S ----------------------
// Forward declarations - use these instead whenever possible...

// ----- C L A S S :   S e c t i o n E m b e d d e d M e t a d a t a ---------

class SectionEmbeddedMetadata : public Section {
 public:
  SectionEmbeddedMetadata();
  virtual ~SectionEmbeddedMetadata();

 private:
  // Purposefully private and undefined ctors...
  SectionEmbeddedMetadata(const SectionEmbeddedMetadata& obj);
  SectionEmbeddedMetadata& operator=(const SectionEmbeddedMetadata& obj);

 private:
  // Static initializer helper class
  static class _init {
   public:
    _init() { registerSectionCtor(EMBEDDED_METADATA, "EMBEDDED_METADATA", "", false, false, boost::factory<SectionEmbeddedMetadata*>()); }
  } _initializer;
};

#endif
