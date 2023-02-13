#ifndef GDEL2D_INPUTCREATOR_H
#define GDEL2D_INPUTCREATOR_H

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include "../inc/PointType.h"
#include "CommonTypes.h"
#include "HashTable.h"
#include "RandGen.h"
#include <pcl/point_cloud.h>

const int GridSize = 8192;

typedef HashTable<Point2, int> PointTable;

enum Distribution
{
    UniformDistribution,
    GaussianDistribution,
    DiskDistribution,
    ThinCircleDistribution,
    CircleDistribution,
    GridDistribution,
    EllipseDistribution,
    TwoLineDistribution
};

const std::string DistStr[] = {"Uniform", "Gaussian", "Disk", "ThinCircle", "Circle", "Grid", "Ellipse", "TwoLines"};

struct InputCreatorPara
{
    bool         _inFile = false;
    std::string  _inFilename;
    std::string  _inConstraintFilename;
    int          _pointNum      = 1000;
    Distribution _dist          = UniformDistribution;
    int          _seed          = 76213898;
    int          _constraintNum = -1;
    bool         _saveToFile    = false;
    std::string  _savePath;
};

class InputCreator
{
  private:
    RandGen     randGen;
    Point2HVec  inPointVec;
    SegmentHVec inConstraintVec;

    void randCirclePoint(double &x, double &y);

    void makePoints(int pointNum, Distribution dist, Point2HVec &pointVec, int seed = 0);

    void readPoints(const std::string &inFilename);

    void readPoints(const std::string &inFilename, const pcl::PointCloud<POINT_TYPE>::Ptr &InputPC);

    void readConstraints(const std::string &inFilename);

  public:
    InputCreator() = default;

    void createPoints(const InputCreatorPara &InputPara, Point2HVec &pointVec, SegmentHVec &constraintVec);

    void createPoints(const InputCreatorPara &InputPara, const pcl::PointCloud<POINT_TYPE>::Ptr &InputPC, Point2HVec &pointVec, SegmentHVec &constraintVec);
};

#endif //GDEL2D_INPUTCREATOR_H