#ifndef GDEL2D_EDIT_POINTTYPE_H
#define GDEL2D_EDIT_POINTTYPE_H

//#define DISABLE_PCL_INPUT

#ifndef DISABLE_PCL_INPUT

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <pcl/memory.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>

#ifndef POINT_TYPE
#define POINT_TYPE tTrimblePoint
#endif

struct EIGEN_ALIGN16 tTrimblePoint
{
    PCL_ADD_POINT4D;
    double intensity;
    double returnnumber;
    double numberofreturns;
    double scandirectionflag;
    double edgeofflightline;
    double classification;
    double scananglerank;
    double userdata;
    double pointsourceid;
    double gpstime;
    double scanchannel;
    double classflags;
    double red;
    double green;
    double blue;
    PCL_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    tTrimblePoint,
    (float, x, x)(float, y, y)(float, z, z)(double, intensity, intensity)(double, returnnumber, returnnumber)(
        double,
        numberofreturns,
        numberofreturns)(double, scandirectionflag, scandirectionflag)(double, edgeofflightline, edgeofflightline)(
        double,
        classification,
        classification)(double, scananglerank, scananglerank)(double, userdata, userdata)(double,
                                                                                          pointsourceid,
                                                                                          pointsourceid)(double,
                                                                                                         gpstime,
                                                                                                         gpstime)(
        double,
        scanchannel,
        scanchannel)(double, classflags, classflags)(double, red, red)(double, green, green)(double, blue, blue))

#endif
#endif //GDEL2D_EDIT_POINTTYPE_H
