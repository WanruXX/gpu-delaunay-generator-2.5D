#ifndef DELAUNAY_GENERATOR_DELAUNAYCHECKER_H
#define DELAUNAY_GENERATOR_DELAUNAYCHECKER_H

#include "CPU/PredWrapper.h"
#include "GPU/GpuDelaunay.h"

class DelaunayChecker
{
  private:
    const Input        &input;
    Output       &output;

    PredWrapper predWrapper;

    size_t getVertexCount() const;
    size_t getSegmentCount() const;
    size_t getTriangleCount();

  public:
    DelaunayChecker() = delete;
    DelaunayChecker(const Input &inputRef, Output &outputRef);
    void        checkEuler();
    void        checkAdjacency() const;
    void        checkOrientation();
    void        checkDelaunay();
    void        checkConstraints();
};

#endif //DELAUNAY_GENERATOR_DELAUNAYCHECKER_H
