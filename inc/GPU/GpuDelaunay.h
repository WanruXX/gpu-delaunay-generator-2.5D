#ifndef DELAUNAY_GENERATOR_GPUDELAUNAYC_H
#define DELAUNAY_GENERATOR_GPUDELAUNAYC_H

#include <array>
#include <iomanip>
#include <iostream>
#include "../IOType.h"
#include "../PerfTimer.h"
#include "CudaWrapper.h"
#include "DPredWrapper.h"
#include "HostToKernel.h"
#include "SmallCounters.h"

////
// Consts
////

constexpr int BlocksPerGrid       = 512;
constexpr int ThreadsPerBlock     = 128;
constexpr int PredBlocksPerGrid   = 64;
constexpr int PredThreadsPerBlock = 32;

////
// Main class
////

constexpr int TriSegNum            = 3;
constexpr int TriSeg[TriSegNum][2] = {{0, 1}, {1, 2}, {2, 0}};

class GpuDel
{
private:
    const Input  *_input  = nullptr;
    Output       *_output = nullptr;

    // Input
    PointDVec   _pointVec;
    EdgeDVec    _constraintVec;
    int         _pointNum = 0;
    int         _triMax   = 0;
    double      _minVal   = 0;
    double      _maxVal   = 0;

    // Output - Size proportional to triNum
    TriDVec    _triVec;
    TriOppDVec _oppVec;
    CharDVec   _triInfoVec;

    // State
    bool       _doFlipping = false;
    ActTriMode _actTriMode = ActTriMarkCompact;
    int        _insNum     = 0;

    // Supplemental arrays - Size proportional to triNum
    IntDVec  _actTriVec;
    Int2DVec _triMsgVec;
    FlipDVec _flipVec;
    IntDVec  _triConsVec;
    IntDVec  _actConsVec;

    MemoryPool _memPool;

    // Supplemental arrays - Size proportional to vertNum
    IntDVec _orgPointIdx;
    IntDVec _vertTriVec;

    // Very small
    IntHVec       _orgFlipNum;
    SmallCounters _counters{};
    Point         _ptInfty;
    int           _infIdx     = 0;
    int           _availPtNum = 0;
    DPredWrapper  _dPredWrapper;

    // Diagnostic - Only used when enabled
    IntDVec _circleCountVec;
    IntDVec _rejFlipVec;

    Diagnostic  _diagLogCompact, _diagLogCollect;
    Diagnostic *_diagLog = nullptr;

    IntHVec _numActiveVec;
    IntHVec _numFlipVec;
    IntHVec _numCircleVec;

    RealHVec _timeCheckVec;
    RealHVec _timeFlipVec;

    // Timing
    CudaTimer _profTimer[ProfLevelCount];

private:
    // Helpers
    void constructInitialTriangles();
    void markSpecialTris();
    void expandTri(int newTriNum);
    void splitTri();
    void initProfiling();
    void doFlippingLoop(CheckDelaunayMode checkMode);
    bool doFlipping(CheckDelaunayMode checkMode);
    void shiftTri(IntDVec &triToVert, IntDVec &splitTriVec);
    void relocateAll();

    void startTiming(ProfLevel level);
    void stopTiming(ProfLevel level, double &accuTime);
    void pauseTiming(ProfLevel level);
    void restartTiming(ProfLevel level, double &accuTime);
    void shiftOppVec(IntDVec &shiftVec, TriOppDVec &dataVec, int size);
    void compactTris();
    void dispatchCheckDelaunay(CheckDelaunayMode checkMode, int orgActNum, IntDVec &triVoteVec);

    template <typename T>
    void shiftExpandVec(IntDVec &shiftVec, DevVector<T> &dataVec, int size);

    void initForConstraintInsertion();
    bool markIntersections();
    bool doConsFlipping(int &flipNum);
    void updatePairStatus();
    void checkConsFlipping(IntDVec &triVoteVec);

    // Main
    void initForFlip();
    void splitAndFlip();
    void doInsertConstraints();
    void outputToHost();
    void cleanup();

public:
    GpuDel() = default;

    static void getEdges(const Tri &t, Edge *sArr);

    void compute(const Input &input, Output &output);
};

//constexpr int GpuDel::TriSegNum;
//constexpr int GpuDel::TriSeg[3][2];

#endif //DELAUNAY_GENERATOR_GPUDELAUNAYC_H