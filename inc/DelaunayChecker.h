#ifndef GDEL2D_DELAUNAYCHECKER_H
#define GDEL2D_DELAUNAYCHECKER_H

#include "CPU/PredWrapper.h"
#include "GPU/GpuDelaunay.h"

class DelaunayChecker
{
  private:
    GDel2DInput  &_input;
    GDel2DOutput &_output;

    PredWrapper2D _predWrapper;

    size_t getVertexCount() const;
    size_t getSegmentCount() const;
    size_t getTriangleCount();

  public:
    DelaunayChecker(GDel2DInput &input, GDel2DOutput &output);
    void        checkEuler();
    void        checkAdjacency() const;
    void        checkOrientation();
    void        checkDelaunay();
    void        checkConstraints();
};

#endif //GDEL2D_DELAUNAYCHECKER_H
