#if 0
/*
 * Copyright 2019 Xilinx, Inc.
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

#ifndef __FFT_COMPLEX__
#define __FFT_COMPLEX__

#define __USE_NAME_SPACE__
#undef __USE_NAME_SPACE__

#ifndef __SYNTHESIS__
#include <iostream>
#endif
#ifdef __USE_NAME_SPACE__
namespace xf {
namespace dsp { 
namespace fft {
#endif
  template<typename T_in> class std::complex
  {
    private:
      T_in m_realPart, m_imagPart;

    public:

      std::complex()
      {

      }
      std::complex(const T_in& p_real, const T_in& p_imag)
      {
        m_realPart = p_real;
        m_imagPart = p_imag;
      }
      std::complex(const T_in& p_real)
      {
        m_realPart = p_real;
        m_imagPart = 0;
      }

      inline T_in real() const
      {
        return(m_realPart);
      }

      inline T_in& real()
      {
        return(m_realPart);
      }

      inline void real(const T_in& p_real)
      {
        m_realPart = p_real;
      }

      inline T_in imag() const
      {
        return(m_imagPart);
      }

      inline T_in& imag()
      {
        return(m_imagPart);
      }
      inline void imag(const T_in& p_imag)
      {
        m_imagPart = p_imag;
      }

      inline std::complex<T_in>& operator= (const T_in& rhs)
      {
        m_realPart = rhs;
        m_imagPart = 0;
        return *this;
      }
      template<typename T2>
      inline std::complex<T_in>& operator= (const std::complex<T2>& rhs)
      {
        m_realPart = rhs.real();
        m_imagPart = rhs.imag();
        return *this;
      }

      template<typename T2>
      inline std::complex<T_in>& operator*= (const T2& rhs)
      {
        m_realPart *= rhs;
        m_imagPart *= rhs;
        return *this;
      }
      template<typename T2>
      inline std::complex<T_in>& operator*= (const std::complex<T2>& rhs)
      {
        T_in tmp1 = m_realPart*rhs.real();
        T_in tmp2 = m_imagPart*rhs.imag();
        T_in tmp3 = m_realPart*rhs.imag();
        T_in tmp4 = m_imagPart*rhs.real();
        m_realPart = tmp1 - tmp2;
        m_imagPart = tmp3 + tmp4;
        return *this;
      }

      inline std::complex<T_in>& operator/= (const T_in& rhs)
      {
        m_realPart /= rhs;
        m_imagPart /= rhs;
        return *this;
      }

      inline std::complex<T_in> operator/ (const T_in& rhs)
      {
        std::complex<T_in> tmp(*this);
        tmp /= rhs;
        return tmp;
      }
      inline std::complex<T_in>& operator/=
                                            (
                                             const std::complex<T_in>& rhs
                                            )
      {

        std::complex<T_in> conj = x_conj(rhs);
        std::complex<T_in> a    = (*this)*conj;
        std::complex<T_in> b    = conj*rhs;
        m_realPart = a.real() / b.real();
        m_imagPart = a.imag() / b.real();
        return *this;
      };

      inline std::complex<T_in>& operator+= (const T_in& rhs)
      {
        m_realPart += rhs;
        return *this;
      }

      inline std::complex<T_in> operator+= (const std::complex<T_in>& rhs)
      {
        m_realPart += rhs.real();
        m_imagPart += rhs.imag();
        return *this;
      }


      template<typename T2>
      inline std::complex<T_in> operator+= (const std::complex<T2>& rhs)
      {
        m_realPart += rhs.real(); m_imagPart += rhs.imag(); return *this;
      }

      inline std::complex<T_in> operator+ (const T_in& rhs) {
        std::complex<T_in> tmp;
        tmp = *this;
        tmp += rhs;
        return tmp;
      }

      inline std::complex<T_in> operator+ (const std::complex<T_in>& rhs) {
         std::complex<T_in> tmp ;
         tmp = *this;
         tmp += rhs;
         return tmp;
      }




      inline std::complex<T_in>& operator-= (const T_in& rhs)
      {
        m_realPart -= rhs;
        return *this;
      }

      inline std::complex<T_in>& operator-= (const std::complex<T_in>& rhs)
      {
        m_realPart -= rhs.real();
        m_imagPart -= rhs.imag();
        return *this;
      }

      template<typename T2>
      inline std::complex<T_in>& operator-= (const std::complex<T2>& rhs)
      {

        m_realPart -= rhs.real();
        m_imagPart -= rhs.imag();
        return *this;
      }

      inline std::complex<T_in> operator- (const T_in& rhs)
      {
        std::complex<T_in> tmp;
        tmp = *this;
        tmp -= rhs;
        return tmp;
      }
      inline  std::complex<T_in> operator- (const std::complex<T_in>& rhs)
      {
        std::complex<T_in> tmp ;
        tmp = *this;
        tmp -= rhs;
        return tmp;

      }

      inline std::complex<T_in> operator- ()
      {
        std::complex<T_in> tmp(*this);
        tmp.real(-real());
        tmp.imag(-imag());
        return tmp;
      }

#ifndef __SYNTHESIS__

      friend std::ostream &operator<<(
                                      std::ostream &out,
                                      const std::complex<T_in> &c
                                     )
      {
        bool neg_imag = c.imag() <= -0 ? true : false;
        T_in imag     = neg_imag ? (T_in)-c.imag() : (T_in)c.imag();

        out << c.real() << (neg_imag ? " - j*":" + j*") << imag;
        return out;
      }

#endif

    }; // std::complex

    // Non-member Operator ==, != for std::complex
    template<typename T_in>
    inline bool operator== (const T_in& lhs, const std::complex<T_in>& rhs)
    {
      return (lhs == rhs.real()) && (0 == rhs.imag());
    }
    template<typename T_in>
    inline bool operator== (const std::complex<T_in>& lhs, const T_in& rhs)
    {
      return (lhs.real() == rhs) && (lhs.imag() == 0);
    }
    template<typename T_in>
    inline bool operator== (
                            const std::complex<T_in>& lhs,
                            const std::complex<T_in>& rhs
                           )
    {
      return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
    }

    template<typename T_in>
    inline bool operator!= (const T_in& lhs, const std::complex<T_in>& rhs)
    {
      return (lhs != rhs.real()) || (0 != rhs.imag());
    }

    template<typename T_in>
    inline bool operator!= (const std::complex<T_in>& lhs, const T_in& rhs)
    {
      return (lhs.real() != rhs) || (lhs.imag() != 0);
    }

    template<typename T_in>
    inline bool operator!= (
                            const std::complex<T_in>& lhs,
                            const std::complex<T_in>& rhs
                           )
    {
      return (lhs.real() != rhs.real()) || (lhs.imag() != rhs.imag());
    }
#ifdef __USE_NAME_SPACE__
}//end namespace fft
}//end namespace dsp
}// end namespace xf
#endif

#endif //__FFT_COMPLEX__
#endif
