#ifndef DELAUNAY_GENERATOR_POINTTYPE_H
#define DELAUNAY_GENERATOR_POINTTYPE_H

#ifdef WITH_PCL

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <pcl/memory.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>

// you can also define your customized point type here, for example:
//struct EIGEN_ALIGN16 MyPointType
//{
//    PCL_ADD_POINT4D;
//    float intensity;
//    long  timestamp;
//    PCL_MAKE_ALIGNED_OPERATOR_NEW
//};
//
//POINT_CLOUD_REGISTER_POINT_STRUCT(MyPointType,
//    (float, x, x)
//    (float, y, y)
//    (float, z, z)
//    (float, intensity, intensity)
//    (long,timestamp,timestamp)
//)

using POINT_TYPE = pcl::PointXYZ;
#endif

#endif //DELAUNAY_GENERATOR_POINTTYPE_H