#ifndef GDEL2D_EDIT_TRIANGULATIONHANDLER_H
#define GDEL2D_EDIT_TRIANGULATIONHANDLER_H

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include "DelaunayChecker.h"
#include "GPU/GpuDelaunay.h"
#include "InputCreator.h"
#include "PerfTimer.h"
#include <iomanip>


class TriangulationHandler
{
  private:
    double InitX = 0;
    double InitY = 0;
    double InitZ = 0;

    // Main
    int  _runNum  = 1;
    bool _doCheck = false;

    InputCreatorPara PtCreatorPara;

    bool        _outputResult = false;
    std::string _outCheckFilename;
    std::string _outMeshFilename;

    // In-Out Data
    GDel2DInput  _input;
    GDel2DOutput _output;

    // Statistics
    Statistics statSum;

    pcl::PointCloud<POINT_TYPE>::Ptr InputPC;

    TriangulationHandler() = default;
    void reset();
    void saveResultsToFile() const;

  public:
    explicit TriangulationHandler(const char *InputYAMLFile);
    void run();
};

#endif //GDEL2D_EDIT_TRIANGULATIONHANDLER_H