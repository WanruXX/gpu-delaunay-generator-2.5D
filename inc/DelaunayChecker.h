#ifndef DELAUNAY_GENERATOR_DELAUNAYCHECKER_H
#define DELAUNAY_GENERATOR_DELAUNAYCHECKER_H

#include "CPU/PredWrapper.h"
#include "GPU/GpuDelaunay.h"

class DelaunayChecker
{
  private:
    Input        &_input;
    Output       &_output;

    PredWrapper2D _predWrapper;

    size_t getVertexCount() const;
    size_t getSegmentCount() const;
    size_t getTriangleCount();

  public:
    DelaunayChecker(Input &input, Output &output);
    void        checkEuler();
    void        checkAdjacency() const;
    void        checkOrientation();
    void        checkDelaunay();
    void        checkConstraints();
};

#endif //GDEL2D_DELAUNAYCHECKER_H
