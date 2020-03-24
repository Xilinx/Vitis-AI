#ifndef __COMMON_HPP__
#define __COMMON_HPP__

//#define __XCODE__

#ifdef _WIN32

#define __IMAGE_FORMAT__ ".jpg"
#define __CVLOADIMAGE_WORKING__

#else
// __CVLOADIMAGE_WORKING__ should not be defined for ".jpg" format
#define __IMAGE_FORMAT__ ".jpg"
//#define __IMAGE_FORMAT__ ".ppm"
#define __CVLOADIMAGE_WORKING__

#endif

#endif
