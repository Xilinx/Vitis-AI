// g++ -std=c++11 `python3 -m pybind11 --includes` -rdynamic -Wl,-rpath,${CONDA_PREFIX}/lib call.cpp AksDataDescriptor.cpp -o embd.exe $CONDA_PREFIX/lib/libpython3.6m.so

#ifndef __AKS_EMBED_PY_H_
#define __AKS_EMBED_PY_H_

#include <iostream>
#include <pybind11/numpy.h>
using namespace AKS;

// type caster: DD* <-> NumPy-array
// TODO : Does it really have to be DD*, or just DD is sufficient?
// TODO : Better expose DataDescriptor so that explicit data copy could be avoided
namespace pybind11 { namespace detail {
  template <> struct type_caster<DataDescriptor> : public type_caster_base<DataDescriptor> {
    using base = type_caster_base<DataDescriptor>;
    public:
      // PYBIND11_TYPE_CASTER(DataDescriptor*, _("DataDescriptor*"));

      // Conversion part 1 (Python -> C++)
      bool load(py::handle src, bool convert)
      {
        // std::cout << "Calling load DD*" << std::endl;
        if (!convert && !py::array_t<float>::check_(src))
          return false;

        auto buf    = py::array_t<float>::ensure(src);
        if (!buf)
          return false;

        auto dims   = buf.ndim();
        if (dims < 1)
          return false;

        std::vector<int> shape(buf.ndim());

        for ( int i = 0 ; i<buf.ndim() ; i++ )
          shape[i]  = buf.shape()[i];

        DataDescriptor* tmp = new DataDescriptor(shape,DataType::FLOAT32);
        value = static_cast<void*>(tmp);

        memcpy(tmp->data(), buf.data(), tmp->getNumberOfElements() * sizeof(float));
        return true;
      }

      //Conversion part 2 (C++ -> Python)
      static py::handle cast(const DataDescriptor* src,
        py::return_value_policy policy, py::handle parent)
      {
        // std::cout << "Calling cast DD*" << std::endl;
        py::array a(src->getShape(), src->getStride(), (float*)src->const_data<float>() );
        return a.release();
      }
  };
}}  // namespace pybind11 { namespace detail {

#endif
