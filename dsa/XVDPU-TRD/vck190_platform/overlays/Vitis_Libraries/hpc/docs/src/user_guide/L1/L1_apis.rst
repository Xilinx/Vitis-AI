.. 
   Copyright 2019 - 2021 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _user_guide_overview_l1_rtm_api:

******************
L1 APIs
******************


MLP
===========

The basic components of the FCN are defined in the template class FCN. Frequent
used activation functions are also implemented.

.. toctree::
   :maxdepth: 2

.. include:: namespace_xf_hpc_mlp.rst

CG Solver
===========

Some basic CG components are defined under the namespace **cg** 

.. toctree::
   :maxdepth: 2

.. include:: namespace_xf_hpc_cg.rst

Reverse Time Migration
==================================
Here describes the basic components for solving wave equations by explicit FDTD method, and components for RTM. 
These components are further classified according to the
problems' dimentions e.g. 2D or 3D. 

Data Movers
---------------------------------
Data movers are implemented for kernel memory communication, dataType conversion and stream datawidth
conversion. 

.. toctree::
   :maxdepth: 2

.. include:: namespace_xf_hpc_rtm.rst

