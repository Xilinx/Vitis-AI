/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
/*
 * Filename: ndarray.hpp
 *
 * Description:
 * This network is used to detecting objects from a input image.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <string>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/profiling.hpp>

template <class T>
class Ndarray {
private:
  size_t capacity=0;
  // std::vector<T> data;
public:
  std::vector<T> data;
  std::vector<size_t> shape;
  size_t dim=0;

  Ndarray() = default;


//tested
  Ndarray(const std::vector<T>& init_data, const std::vector<size_t> &shapes) {
    zeros(shapes);
    data = init_data;
  }


//tested
  void zeros(const std::vector<size_t> &shapes) {
    shape = shapes;
    cal_dim();
    cal_size();
    data.resize(capacity, T(0));
  }

  void arange(const int size) {
    shape.push_back(size);
    cal_dim();
    cal_size();
    data.resize(size);
    for (auto i = 0; i < size; i++)
      data[i] = T(i);
  }

  void ones(const std::vector<size_t> &shapes) {
    shape = shapes;
    cal_dim();
    cal_size();
    data.resize(capacity, 1);
  }

  Ndarray<T> copy() const {
    Ndarray<T> B;
    B.zeros(shape);
    std::copy_n(this->get_data(), this->size(), B.mutable_data());
    
    return B;
  }

//un
  void swap_data(const std::vector<T>& input_data) {
    if(input_data.size() != capacity) {
      std::cerr << "input size != capacity" << std::endl;
    }
    data.swap(input_data);
  }

  const T* get_data() const{
    return data.data();
  } 

  T* mutable_data() {
    return data.data();
  } 

  void cal_size() {
    check_shape(); 
    capacity = shape[0];
    for (auto i = 1u; i < shape.size(); i++)
      capacity *= shape[i];
  }

  void cal_dim() {
    dim = shape.size();
    if (dim == 0) {
      std::cerr << "The dimensions need to >=1, but now input shape is 0" << std::endl;
      abort();
    }
  }

  void check_shape() {
    for (auto &i : shape) {
      if (i < 0) {
        std::cerr << "The shape need to >0 " << std::endl; 
        abort();
      }
    }
  }

  size_t size() const {
    return capacity;
  }

  size_t get_dim() const {
    return dim;
  }

  //un
  void reshape(const std::vector<size_t> &shapes) {
    check_shape();
    auto new_size = 1u;
    for (auto i = 0u; i < shapes.size(); i++)
      new_size *= shapes[i];
    if (new_size != capacity) {
      std::cerr << "reshape: new size != old size"  << new_size << " " << capacity <<  std::endl;
      abort();
    }
    shape = shapes;
    cal_dim();
  }

  void resize(size_t new_size) {
    data.resize(new_size);
    capacity = new_size;
  } 

  T &operator[](const size_t ind)  {
    if (ind < capacity) {
      return this->data[ind]; 
    } else {
      std::cerr  << ind << "[] Out of range" << std::endl;
      exit(-1);
    }
  }

  const T operator[](const size_t ind) const {
    if (ind < capacity) {
      return this->data[ind]; 
    } else {
      std::cerr  << ind << "[] Out of range" << std::endl;
      exit(-1);
    }
  }
  void print(std::string info="") {
    std::cout << info << " dim: " << dim << ", size: " << capacity << std::endl << "shape: [";
    for (auto &sp : shape)
      std::cout << sp << " ";
    std::cout << "]" << std::endl;
  }

  void mul(std::vector<size_t> range, T value) {
   for (auto i = range[0]; i < range[1]; i++)  {
     data[i] *= value;
   }
  }
  void mul(T value) {
    for (auto i = 0u; i < capacity; i++) {
      data[i] *= value;
    }
  }

  typename std::vector<T>::iterator mutable_from_begin() {
    return this->data.begin();
  }

  Ndarray<int> filter(float thr) {
    Ndarray<int> fil;
    fil.zeros(this->shape);
    std::transform(this->data.begin(), this->data.end(), fil.mutable_from_begin(), 
       [&thr](T value)->int{return value > thr ? 1 : 0;});     
    return std::move(fil);
  }
  
  inline __attribute__((always_inline)) Ndarray<T> bool_select(const Ndarray<int>& A) {
    std::vector<T> B;
    for (auto i = 0u; i < capacity; i++) {
      auto temp = A[i];
      if (temp == 1) 
        B.push_back(data[i]);
    }
    Ndarray<T> C(B, {B.size()});
    return std::move(C); 
  }

  inline __attribute__((always_inline)) std::vector<T> select(const std::vector<size_t>& A) {
    std::vector<T> B(A.size());
    for (auto i = 0u; i < A.size(); i++) {
      B[i] = data[A[i]];
    }
    return std::move(B);
  }
  
  template<typename _T>
  inline __attribute__((always_inline)) Ndarray<T> select(const Ndarray<_T>& A) {
    std::vector<T> B(A.size());
    for (auto i = 0u; i < A.size(); i++) {
      B[i] = data[A[i]];
    }
    Ndarray<T> C(B, {B.size()});
    return std::move(C);
  }
  
  Ndarray<int> nonzero() {
    std::vector<int> coord;
    auto shape_0 = 0u;
    for (auto ind = 0u; ind < capacity; ind++) {
      if (data[ind] != 0) {
        auto remaining_ind = ind;
        shape_0++;
        for (auto dim = 1u; dim < this->shape.size(); dim++) {
          coord.push_back(remaining_ind / this->shape[dim]);
          remaining_ind = remaining_ind % this->shape[dim];
        }
        coord.push_back(remaining_ind);
      }
    } 
    Ndarray<int> B(coord, {shape_0, this->shape.size()});
    return std::move(B);
  }

  Ndarray<T> index_select(size_t d, const Ndarray<int>& index)  { // only support 1d or 2d
    std::vector<T> A;
    if (d == 0) {
      if (this->dim <= 1) {
        A.reserve(index.size()); // size = select size per row * rows
        for (auto j = 0u; j < index.size(); j++) {
          A.emplace_back(data[index[j]]);
        }
      } else {
          LOG(FATAL) << "index select: not support dim"<< std::endl; 
      } 
    } else {
      LOG(FATAL) << "index select: dim: " << d << " not support now"<< std::endl; 
    } 
    auto new_shape = shape;
    new_shape[d] = index.size();
    Ndarray<T> B(A, new_shape);
    //todo
    return std::move(B);
  }  

  Ndarray<T> slice(std::vector<std::vector<size_t>> range) {
    std::vector<size_t> inds_iter{0};
    std::vector<size_t> new_shape;
    if (dim != range.size()) {
      LOG(FATAL) << "slice: out of range " << std::endl;
    }
    for(auto i = 0u; i < dim; i++) {
      std::vector<size_t> temp;
      for (auto k = 0u; k < inds_iter.size(); k++) {
        for (auto j = range[i][0]; j < range[i][1]; j++) {
          // std::cout << "23 " << inds_iter[k] * shape[i] + j << std::endl;
          // std::cout << i << " " << k << " " << j << std::endl;
          temp.push_back(inds_iter[k] * shape[i] + j); 
        }
      }
      inds_iter = temp; 
    }
    for (auto & i : range) {
      //auto s = 0u;
      new_shape.push_back(i[1] - i[0]);
    }
    //std::cout << "666 " << inds_iter.size() << std::endl;
    //for (auto i = 0u; i < inds_iter.size(); i++) {
      //std::cout << inds_iter[i] << std::endl;
    //}    
    auto slice_data = select(inds_iter);
    //std::cout << "size " << slice_data.size() << std::endl;
    Ndarray<T> B(slice_data, new_shape);
    return std::move(B);
  }

  //something todo
  void extend(Ndarray<T> B) {
    this->shape[0] += B.shape[0];
    //this->print();
    auto old_capacity = capacity;
    cal_size();
    this->data.resize(capacity);
    std::copy_n(B.mutable_from_begin(), B.size(), data.begin() + old_capacity);
  }


  template<typename _T>
  Ndarray<T> slice_dim0(const Ndarray<_T>& range, bool select_by_bool = false) {
    //print();
    auto slice_shape = shape;
    Ndarray<T> B;
    slice_shape[0] = 0;
    B.zeros(slice_shape);
    std::vector<std::vector<size_t>> slice_vec;
    for(auto i = 1u; i < dim; i++) {
      slice_vec.push_back({0, shape[i]});
      //print();
      //std::cout << this->shape[i] << " " << dim  << std::endl;
    }
    if(select_by_bool == true) {
      std::vector<_T> temp;
      for (auto i = 0u; i < range.size(); i++) {
        if(range[i] == 1) {
          temp.push_back(i);
        }
      }
      Ndarray<_T> temp_array(temp, {temp.size()});
    for (auto i = 0u; i < temp_array.size(); i++) {
      std::vector<std::vector<size_t>> temp_vec = {{(size_t)temp_array[i], (size_t)temp_array[i]+1}};
      for(auto & j:slice_vec) {
        temp_vec.push_back(j);
      }
      auto A = slice(temp_vec);
      B.extend(A);
    }
    return B;
    }
    for (auto i = 0u; i < range.size(); i++) {
      std::vector<std::vector<size_t>> temp_vec = {{(size_t)range[i], (size_t)range[i]+1}};
      for(auto & j:slice_vec) {
        temp_vec.push_back(j);
      }
      auto A = slice(temp_vec);
      B.extend(A);
    }
    return B;
  }


  template<typename _T>
  Ndarray<_T> reduce_todim1(_T dtype) {
    auto reduce_len = shape[0]; 
    auto reduce_step = capacity / reduce_len;
    std::vector<_T> result;
    // std::cout << "reduce " << reduce_len << " " << reduce_step << std::endl;
    for(auto i = 0u; i < reduce_len; i++) {
      _T temp = _T(0);
      for(auto j = i * reduce_step; j < (i + 1) * reduce_step; j++) {
        temp += (_T)data[j];
      }
      result.push_back(temp);
    }
    Ndarray<_T> B(result, {reduce_len});
    return std::move(B);
  }

  Ndarray<size_t> argsort(int ascending=0) {
    Ndarray<size_t> B;
    B.zeros(shape);
	  for (auto i = 0u; i < capacity; ++i)
		  B.data[i] = i;
    
    if (ascending==0) {
	    std::sort(B.data.begin(), B.data.end(),
		    [&](int pos1, int pos2) {return (this->data[pos1] < this->data[pos2]);});
    }else {
	    std::sort(B.data.begin(), B.data.end(),
		    [&](int pos1, int pos2) {return (this->data[pos1] > this->data[pos2]);});
    }
	  return std::move(B);
  }

  Ndarray<T> transpose2d() {
    if(dim!=2) {
      LOG(FATAL) << "trasnpose: only suppport 2D matrix" << std::endl;
    }
    Ndarray<T> B;
    B.zeros({shape[1], shape[0]});
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat1(data.data(), shape[0], shape[1]);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat2(B.mutable_data(), shape[1], shape[0]);
    //std::cout << mat1 << std::endl;
    mat2 = mat1.transpose();
    //std::cout << mat2 << std::endl;
    return std::move(B);
  }

  Ndarray<T> expand(size_t expand_size) {
    Ndarray<T> B;
    B.zeros({expand_size, capacity});
    for(auto i = 0u; i < expand_size; i++) {
      copy_n(data.begin(), capacity, B.mutable_data() + i*capacity); 
    }
    return std::move(B);
  }

  Ndarray<float> _float() {
    Ndarray<float> B;
    B.zeros(shape);
    for (auto i = 0u; i < capacity; i++) {
      B[i] = (float)data[i];
    }
    return std::move(B);
  } 

  Ndarray<T> triu(int diagonal) {
    Ndarray<T> B;
    B.zeros(shape);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat1(data.data(), shape[0], shape[1]);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat2(B.mutable_data(),shape[0], shape[1]);
    if(diagonal == 1) {
      mat2 = mat1.template triangularView<Eigen::StrictlyUpper>();    
      return std::move(B);
    } else {
      LOG(FATAL) << "triu: only support diagonal=1" << std::endl;
      exit(-1);
    }
  }

  Ndarray<T> max(size_t dim = 0) {
    if(dim !=0) {
      std::cerr << "max: only support dim = 0" << std::endl;
      exit(-1);
    }
    Ndarray<T> B;
    B.zeros({shape[1]});
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat1(data.data(), shape[0], shape[1]);
    Eigen::Map<Eigen::Matrix<T, -1, 1>> mat2(B.mutable_data(),shape[0]);
    //std::cout << mat1 << std::endl;
    mat2 = mat1.colwise().maxCoeff();
    //std::cout << mat2 << std::endl;
    return std::move(B); 

  }

  Ndarray<T> min(size_t dim = 0) {
    if(dim !=0) {
      std::cerr << "max: only support dim = 0" << std::endl;
      exit(-1);
    }
    Ndarray<T> B;
    B.zeros({shape[1]});
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat1(data.data(), shape[0], shape[1]);
    Eigen::Map<Eigen::Matrix<T, -1, 1>> mat2(B.mutable_data(),shape[0]);
    //std::cout << mat1 << std::endl;
    mat2 = mat1.colwise().minCoeff();
    //std::cout << mat2 << std::endl;
    return std::move(B); 
  }


  Ndarray<T> sum(size_t dim) {
    if(dim !=1) {
      std::cerr << "max: only support dim = 1" << std::endl;
      exit(-1);
    }
    Ndarray<T> B;
    B.zeros({shape[1]});
    LOG(INFO) << shape[0] << " " << shape[1];
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> mat1(data.data(), shape[0], shape[1]);
    Eigen::Map<Eigen::Matrix<T, 1, -1>> mat2(B.mutable_data(),shape[0]);
    mat2 = mat1.rowwise().sum();
    std::cout << mat2 << std::endl;
    return std::move(B); 
  }

  T sum() {
    Eigen::Map<Eigen::Matrix<T, -1, 1>> mat1(data.data(), capacity);
    auto b = mat1.sum();
    return b;
  }

  Ndarray<T> pow(size_t n) {
    Ndarray<T> B;
    B.zeros(shape);
    for(auto i = 0u; i < capacity; i++) {
      B[i] = std::pow(data[i], n);
    }
    return std::move(B);
  }



};
//128   128   1

template<typename T>
Ndarray<T> mm(Ndarray<T>& A, Ndarray<T>& B){
    Ndarray<T> C;
    C.zeros({A.shape[0], B.shape[1]});
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat1(A.mutable_data(), A.shape[0], A.shape[1]);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat2(B.mutable_data(), B.shape[0], B.shape[1]);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat3(C.mutable_data(), A.shape[0], B.shape[1]);
    // std::cout << mat1 << std::endl;
    // std::cout << mat2 << std::endl;
    mat3.noalias() = mat1 * mat2; 
    // std::cout << mat3  << std::endl;
    return std::move(C);
}



template<typename T>
Ndarray<T> operator+ (Ndarray<T> A, Ndarray<T> B) {
  Ndarray<T> C;
  C.zeros(A.shape);
  /*
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = A[i] + B[i];
  } 
  */ 
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat1(A.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat2(B.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat3(C.mutable_data(), A.size());
  mat3 =  mat1.array() + mat2.array(); 
  return std::move(C);
}

template<typename T>
Ndarray<T> operator+ (Ndarray<T> A, T B) {
  Ndarray<T> C;
  C.zeros(A.shape);
  /*
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = A[i] + B;
  }  
  */
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat1(A.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat3(C.mutable_data(), A.size());
  mat3 =  mat1.array() + B; 
  return std::move(C);
}

template<typename T>
Ndarray<T> operator * (Ndarray<T> A, Ndarray<T> B) {
  
  Ndarray<T> C;
  C.zeros(A.shape);
  /*
  for (auto i = 0u; i < A.size(); i++) {
    C.get_data()[i] = A[i] * B[i];
  } 
  */ 
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat1(A.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat2(B.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat3(C.mutable_data(), A.size());
  mat3 =  mat1.array() *mat2.array(); 
  return std::move(C);
}

template<typename T>
Ndarray<T> operator * (Ndarray<T> A, T B) {
  Ndarray<T> C;
  C.zeros(A.shape);
  /*
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = A[i] * B;
  } 
  */ 
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat1(A.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat3(C.mutable_data(), A.size());
  mat3 =  mat1.array() * B; 
  return C;
}

template<typename T>
Ndarray<T> operator * (T A, Ndarray<T> B) {
  Ndarray<T> C;
  C.zeros(B.shape);
  /*
  for (auto i = 0u; i < B.size(); i++) {
    C[i] = A * B[i];
  } 
  */
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat1(B.mutable_data(), B.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat3(C.mutable_data(), B.size());
  mat3 =  mat1.array() * A; 
  return C;
}

template<typename T>
Ndarray<T> operator - (Ndarray<T> A, Ndarray<T> B) {
  Ndarray<T> C;
  C.zeros(A.shape);
  /*
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = A[i] - B[i];
  }  
  */
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat1(A.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat2(B.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat3(C.mutable_data(), A.size());
  mat3 =  mat1.array() - mat2.array(); 
  return C;
}

template<typename T>
Ndarray<T> operator - (Ndarray<T> A, T B) {
  Ndarray<T> C;
  C.zeros(A.shape);
  /*
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = A[i] - B;
  }  
  */
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat1(B.mutable_data(), B.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat3(C.mutable_data(), B.size());
  mat3 =  mat1.array() - B; 
  return C;
}

template<typename T>
Ndarray<T> operator / (Ndarray<T> A, Ndarray<T> B) {
  Ndarray<T> C;
  C.zeros(A.shape);
  /*
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = A[i] / B[i];
  } 
  */ 
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat1(A.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat2(B.mutable_data(), A.size());
  Eigen::Map<Eigen::Matrix<T, 1, -1>> mat3(C.mutable_data(), A.size());
  mat3 =  mat1.array() / mat2.array(); 
  return C;
}


template<typename T1,typename T2>
Ndarray<T1> operator % (const Ndarray<T1>& A, const Ndarray<T2>& B) {
  Ndarray<T1> C;
  C.zeros(A.shape);
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = A[i] % B[i];
  }  
  return C;
}

template<typename T1,typename T2>
Ndarray<int> operator > (const Ndarray<T1>& A, const Ndarray<T2>& B) {
  Ndarray<int> C;
  C.zeros(A.shape);
  for (auto i = 0u; i < A.size(); i++) {
    int value = A[i] > (T1)B[i] ? 1 : 0;
    C[i] = value;
  }  
  return std::move(C);
}

template<typename T1,typename T2>
Ndarray<int> operator < (const Ndarray<T1>& A, const Ndarray<T2>& B) {
  Ndarray<int> C;
  C.zeros(A.shape);
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = A[i] < B[i] ? 1 : 0;
  }  
  return std::move(C);
}

template<typename T1,typename T2>
Ndarray<int> operator == (const Ndarray<T1>& A, const Ndarray<T2>& B) {
  Ndarray<int> C;
  C.zeros(A.shape);
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = A[i] == B[i] ? 1 : 0;
  }  
  return std::move(C);
}

template<typename T1,typename T2>
static inline Ndarray<T1> floor_div(const Ndarray<T1>& A, const Ndarray<T2>& B) {
  Ndarray<T1> C;
  C.zeros(A.shape);
  for (auto i = 0u; i < A.size(); i++) {
    C[i] = std::floor(A[i] % B[i]);
  }  
  return std::move(C);
}

template<typename T>
static inline Ndarray<T> _exp(Ndarray<T> A) {
  Ndarray<T> B;
  B.zeros(A.shape);
  for(auto i = 0u; i < A.size(); i++) {
    B[i] = exp(A[i]);
  }
  return std::move(B);
}

static inline Ndarray<float> interpolate(Ndarray<float>& A, size_t _h, size_t _w) {
  if (A.get_dim()== 2) {
    A.reshape({1, A.shape[0], A.shape[1]});
  }
  //A.shape[0] = 1;
  auto old_step = A.shape[1]*A.shape[2];
  //LOG(INFO) << A.shape[1] <<" " <<A.shape[2];
  auto new_step = _h*_w;
  Ndarray<float> B;
  B.zeros({A.shape[0], _h, _w});
  for (auto i = 0u; i < A.shape[0]; i++) {
    cv::Mat ori(A.shape[1], A.shape[2], CV_32FC1, A.mutable_data() + i * old_step);
    cv::Mat dst(_h, _w, CV_32FC1, B.mutable_data() + i* new_step);
    cv::resize(ori,dst, cv::Size(_w, _h));
  }
  return std::move(B);
}

static inline Ndarray<int> interpolate(Ndarray<int>& A, size_t _h, size_t _w) {
  if (A.get_dim()== 2) {
    A.reshape({1, A.shape[0], A.shape[1]});
  }
  //A.shape[0] = 1;
  auto old_step = A.shape[1]*A.shape[2];
  //LOG(INFO) << A.shape[1] <<" " <<A.shape[2];
  auto new_step = _h*_w;
  Ndarray<int> B;
  B.zeros({A.shape[0], _h, _w});
  for (auto i = 0u; i < A.shape[0]; i++) {
    cv::Mat ori(A.shape[1], A.shape[2], CV_32SC1, A.mutable_data() + i * old_step);
    cv::Mat dst(_h, _w, CV_32SC1, B.mutable_data() + i* new_step);
    cv::resize(ori,dst, cv::Size(_w, _h));
  }
  return std::move(B);
}
/*
Ndarray<float> slice(Ndarray<float>, uint axis, uint start, uint end) {
}

Ndarray<int> slice(Ndarray<int>, uint axis, uint start, uint end) {
}

Ndarray<uint8_t> slice(Ndarray<uint8_t>, uint axis, uint start, uint end) {
}
*/
/*
Ndarray<float> slice(Ndarray<float>, uint axis, uint start, uint end) {
}

Ndarray<int> slice(Ndarray<int>, uint axis, uint start, uint end) {
}

Ndarray<uint8_t> slice(Ndarray<uint8_t>, uint axis, uint start, uint end) {
}
*/

