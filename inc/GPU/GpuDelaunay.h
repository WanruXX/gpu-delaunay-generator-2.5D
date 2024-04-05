#ifndef DELAUNAY_GENERATOR_GPUDELAUNAYC_H
#define DELAUNAY_GENERATOR_GPUDELAUNAYC_H

#include "../IOType.h"
#include "../PerfTimer.h"
#include "CudaWrapper.h"
#include "DPredWrapper.h"
#include "HostToKernel.h"
#include "SmallCounters.h"
#include <array>
#include <iomanip>
#include <iostream>

constexpr int BlocksPerGrid        = 512;
constexpr int ThreadsPerBlock      = 128;
constexpr int PredBlocksPerGrid    = 64;
constexpr int PredThreadsPerBlock  = 32;
constexpr int TriSegNum            = 3;
constexpr int TriSeg[TriSegNum][2] = {{0, 1}, {1, 2}, {2, 0}};

class GpuDel
{
  private:
    const Input *inputPtr  = nullptr;
    Output      *outputPtr = nullptr;

    int pointNum = 0;

    PointDVec pointVec;
    EdgeDVec  constraintVec;
    IntDVec   actConsVec;

    int        triMaxNum = 0;
    TriDVec    triVec;
    TriOppDVec oppVec;
    CharDVec   triInfoVec;

    double minVal = 0;
    double maxVal = 0;

    int  infIdx       = 0;
    bool doFlipping   = false;
    int  availPtNum   = 0;
    int  insertTriNum = 0;

    MemoryPool memPool;
    FlipDVec   flipVec;
    Int2DVec   triMsgVec;
    IntDVec    actTriVec;
    IntDVec    triConsVec;

    IntDVec originalPointIdx;
    IntDVec vertexTriVec;
    IntHVec originalFlipNum;

    SmallCounters counters;
    DPredWrapper  dPredWrapper;

    ActTriMode  actTriMode = ActTriMarkCompact;
    Diagnostic  diagLogCompact, diagLogCollect;
    Diagnostic *diagLogPtr = nullptr;

    IntHVec numActiveVec;
    IntHVec numFlipVec;
    IntHVec numCircleVec;

    CudaTimer profTimer[ProfLevelCount];
    IntDVec   circleCountVec;
    IntDVec   rejFlipVec;
    RealHVec  timeCheckVec;
    RealHVec  timeFlipVec;

  private:
    void initProfiling();

    void initForFlip();
    void constructInitialTriangles();
    Tri  setOutputInfPointAndTriangle();

    void splitAndFlip();
    void splitTri();
    void shiftTri(IntDVec &triToVert, IntDVec &splitTriVec);
    template <typename T>
    void shiftExpandVec(IntDVec &shiftVec, DevVector<T> &dataVec, int size);
    void shiftOppVec(IntDVec &shiftVec, TriOppDVec &dataVec, int size);
    void expandTri(int newTriNum);
    void flipLoop(CheckDelaunayMode checkMode);
    bool flip(CheckDelaunayMode checkMode);
    void dispatchCheckDelaunay(CheckDelaunayMode checkMode, int orgActNum, IntDVec &triVoteVec);
    void relocateAll();
    void markSpecialTris();
    void insertConstraints();
    void initForConstraintInsertion();
    bool markIntersections();
    bool flipConstraints(int &flipNum);
    void updatePairStatus();
    void checkConsFlipping(IntDVec &triVoteVec);

    void outputToHost();
    void compactTris();

    void cleanup();

    void startTiming(ProfLevel level);
    void stopTiming(ProfLevel level, double &accuTime);
    void pauseTiming(ProfLevel level);
    void restartTiming(ProfLevel level, double &accuTime);

  public:
    GpuDel();

    static void getEdges(const Tri &t, Edge *sArr);

    void compute(const Input &input, Output &output);
};

//constexpr int GpuDel::TriSegNum;
//constexpr int GpuDel::TriSeg[3][2];

#endif //DELAUNAY_GENERATOR_GPUDELAUNAYC_H