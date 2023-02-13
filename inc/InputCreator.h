#ifndef GDEL2D_INPUTCREATOR_H
#define GDEL2D_INPUTCREATOR_H

#include "CommonTypes.h"
#include "GPU/GpuDelaunay.h"
#include "HashTable.h"
#include "RandGen.h"

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
    RandGen    randGen;
    Point2HVec inPointVec;
#ifndef DISABLE_PCL_INPUT
    pcl::PointCloud<POINT_TYPE>::Ptr inPointCloud = nullptr;
#endif
    SegmentHVec inConstraintVec;

    void randCirclePoint(double &x, double &y);

    void makePoints(int pointNum, Distribution dist, Point2HVec &pointVec, int seed = 0);

    void readPoints(const std::string &inFilename);

    void readConstraints(const std::string &inFilename);

  public:
#ifndef DISABLE_PCL_INPUT
    InputCreator() : inPointCloud(new pcl::PointCloud<POINT_TYPE>){};
    void createPoints(const InputCreatorPara      &InputPara,
                      pcl::PointCloud<POINT_TYPE> &InputPointCloud,
                      Point2HVec                  &InputPointVec,
                      SegmentHVec                 &InputConstraintVec);
#else
    InputCreator() = default;
    void createPoints(const InputCreatorPara &InputPara, Point2HVec &InputPointVec, SegmentHVec &InputConstraintVec);
#endif
};

#endif //GDEL2D_INPUTCREATOR_H