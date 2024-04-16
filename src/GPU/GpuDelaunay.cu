#include "../../inc/CPU/PredWrapper.h"
#include "../../inc/GPU/GpuDelaunay.h"
#include "../../inc/GPU/KerCommon.h"
#include "../../inc/GPU/KerDivision.h"
#include "../../inc/GPU/KerPredicates.h"
#include "../../inc/GPU/ThrustWrapper.h"
#include <iostream>
#include <thrust/gather.h>

namespace
{
struct CompareX
{
    __device__ bool operator()(const Point &a, const Point &b) const
    {
        return a._p[0] < b._p[0];
    }
};

struct Get2Ddist
{
    Point  _a;
    double abx, aby;

    Get2Ddist(const Point &a, const Point &b) : _a(a)
    {
        abx = b._p[0] - a._p[0];
        aby = b._p[1] - a._p[1];
    }

    __device__ int operator()(const Point &c)
    {
        double acx = c._p[0] - _a._p[0];
        double acy = c._p[1] - _a._p[1];

        double dist = abx * acy - aby * acx;

        return __float_as_int(fabs((float)dist));
    }
};

template <typename T>
__global__ void kerShift(KerIntArray shiftVec, T *src, T *dest)
{
    for (int idx = getCurThreadIdx(); idx < shiftVec._num; idx += getThreadNum())
    {
        const int shift = shiftVec._arr[idx];

        dest[idx + shift] = src[idx];
    }
}
constexpr int MaxSamplePerTri = 100;
} // namespace

GpuDel::GpuDel()
{
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}

void GpuDel::compute(const Input &input, Output &output)
{
    inputPtr  = &input;
    outputPtr = &output;

    initProfiling();

    startTiming(ProfNone);
    initForFlip();
    splitAndFlip();
    outputToHost();
    stopTiming(ProfNone, outputPtr->stats.totalTime);

    if (inputPtr->isProfiling(ProfDetail))
    {
        std::cout << " FlipCompact time: ";
        diagLogCompact.printTime();
        std::cout << std::endl;
        std::cout << " FlipCollect time: ";
        diagLogCollect.printTime();
        std::cout << std::endl;
    }

    cleanup();
}

void GpuDel::initProfiling()
{
    outputPtr->stats.reset();
    diagLogCompact.reset();
    diagLogCollect.reset();
    numActiveVec.clear();
    numFlipVec.clear();
    timeCheckVec.clear();
    timeFlipVec.clear();
}

void GpuDel::initForFlip()
{
    startTiming(ProfDefault);

    initSizeAndBuffers();
    findMinMax();
    // Sort points along space curve
    if (!inputPtr->noSort)
    {
        sortPoints();
    }
    // Create first upper-lower triangles
    constructInitialTriangles();

    stopTiming(ProfDefault, outputPtr->stats.initTime);
}

void GpuDel::initSizeAndBuffers()
{
    pointNum = static_cast<int>(inputPtr->pointVec.size()) + 1; // Plus the infinity point
    pointVec.resize(pointNum);
    pointVec.copyFromHost(inputPtr->pointVec);
    constraintVec.copyFromHost(inputPtr->constraintVec);
    actConsVec.resize(constraintVec.size());

    triMaxNum = pointNum * 2;
    triVec.resize(triMaxNum);
    oppVec.resize(triMaxNum);
    triInfoVec.resize(triMaxNum);

    infIdx     = pointNum - 1;
    doFlipping = !inputPtr->insAll;
    availPtNum = pointNum - 4;

    counters.init();

    if (inputPtr->isProfiling(ProfDiag))
    {
        circleCountVec.resize(triMaxNum);
        rejFlipVec.resize(triMaxNum);
    }

    // Preallocate some buffers in the pool
    memPool.reserve<FlipItem>(triMaxNum); // flipVec
    memPool.reserve<int2>(triMaxNum);     // triMsgVec
    memPool.reserve<int>(pointNum);       // vertexTriVec
    memPool.reserve<int>(triMaxNum);      // actTriVec
    memPool.reserve<int>(triMaxNum);      // Two more for common use
    memPool.reserve<int>(triMaxNum);      //

    if (constraintVec.size() > 0)
    {
        memPool.reserve<int>(triMaxNum);
    }
}

void GpuDel::findMinMax()
{
    using DRealPtr = thrust::device_ptr<double>;
    DRealPtr                         coords((double *)toKernelPtr(pointVec));
    thrust::pair<DRealPtr, DRealPtr> ret =
        thrust::minmax_element(coords, coords + static_cast<long>(pointVec.size()) * 2);
    minVal = *ret.first;
    maxVal = *ret.second;
    if (inputPtr->isProfiling(ProfDebug))
    {
        std::cout << "minVal = " << minVal << ", maxVal == " << maxVal << std::endl;
    }
}

void GpuDel::sortPoints()
{
    stopTiming(ProfDefault, outputPtr->stats.initTime);
    startTiming(ProfDefault);

    IntDVec valueVec = memPool.allocateAny<int>(pointNum);
    valueVec.resize(pointVec.size());

    originalPointIdx.resize(pointNum);
    thrust::sequence(originalPointIdx.begin(), originalPointIdx.end(), 0);

    thrust_transform_GetMortonNumber(pointVec.begin(), pointVec.end(), valueVec.begin(), minVal, maxVal);

    thrust_sort_by_key(
        valueVec.begin(), valueVec.end(), make_zip_iterator(make_tuple(originalPointIdx.begin(), pointVec.begin())));

    memPool.release(valueVec);

    stopTiming(ProfDefault, outputPtr->stats.sortTime);
    startTiming(ProfDefault);
}

void GpuDel::constructInitialTriangles()
{
    // First, choose two extreme points along the X axis
    typedef PointDVec::DevPtr PointIter;
    Tri                       firstTri = setOutputInfPointAndTriangle();
    triVec.expand(4);
    oppVec.expand(4);
    triInfoVec.expand(4);

    // Put the initial tets at the Inf list
    kerMakeFirstTri<<<1, 1>>>(toKernelPtr(triVec), toKernelPtr(oppVec), toKernelPtr(triInfoVec), firstTri, infIdx);
    CudaCheckError();

    // Locate initial positions of points
    vertexTriVec.resize(pointNum);
    IntDVec exactCheckVec = memPool.allocateAny<int>(pointNum);
    counters.renew();
    kerInitPointLocationFast<<<BlocksPerGrid, ThreadsPerBlock>>>(
        toKernelArray(vertexTriVec), toKernelPtr(exactCheckVec), counters.ptr(), firstTri);
    kerInitPointLocationExact<<<PredBlocksPerGrid, PredThreadsPerBlock>>>(
        toKernelPtr(vertexTriVec), toKernelPtr(exactCheckVec), counters.ptr(), firstTri);
    CudaCheckError();
    memPool.release(exactCheckVec);
}

Tri GpuDel::setOutputInfPointAndTriangle()
{
    const auto  ret     = thrust::minmax_element(pointVec.begin(), pointVec.end(), CompareX());
    auto        v0      = static_cast<int>(ret.first - pointVec.begin());
    auto        v1      = static_cast<int>(ret.second - pointVec.begin());
    const Point p0      = pointVec[v0];
    const Point p1      = pointVec[v1];
    IntDVec     distVec = memPool.allocateAny<int>(pointNum);
    distVec.resize(pointVec.size());
    thrust::transform(pointVec.begin(), pointVec.end(), distVec.begin(), Get2Ddist(p0, p1));
    const auto  v2 = static_cast<int>(thrust::max_element(distVec.begin(), distVec.end()) - distVec.begin());
    const Point p2 = pointVec[v2];
    memPool.release(distVec);

    if (inputPtr->isProfiling(ProfDebug))
    {
        std::cout << "Leftmost: " << v0 << " --> " << p0._p[0] << " " << p0._p[1] << std::endl;
        std::cout << "Rightmost: " << v1 << " --> " << p1._p[0] << " " << p1._p[1] << std::endl;
        std::cout << "Furthest 2D: " << v2 << " --> " << p2._p[0] << " " << p2._p[1] << std::endl;
    }

    // Check to make sure the 4 points are not co-planar
    double ori = orient2dzero(p0._p, p1._p, p2._p);

    if (almost_zero(ori))
    {
        throw(std::runtime_error("Input too degenerated! Points are almost on the same line!"));
    }

    if (ortToOrient(ori) == OrientNeg)
    {
        std::swap(v0, v1);
    }

    // Compute the centroid of v0v1v2v3, to be used as the kernel point.
    outputPtr->infPt._p[0] = (p0._p[0] + p1._p[0] + p2._p[0]) / 3.0;
    outputPtr->infPt._p[1] = (p0._p[1] + p1._p[1] + p2._p[1]) / 3.0;
    outputPtr->infPt._p[2] = (p0._p[2] + p1._p[2] + p2._p[2]) / 3.0;

    pointVec.resize(pointNum);
    pointVec[infIdx] = outputPtr->infPt;

    if (inputPtr->isProfiling(ProfDiag))
    {
        std::cout << "Kernel: " << outputPtr->infPt._p[0] << " " << outputPtr->infPt._p[1] << " "
                  << outputPtr->infPt._p[2] << std::endl;
    }
    dPredWrapper.init(
        toKernelPtr(pointVec), pointNum, inputPtr->noSort ? nullptr : toKernelPtr(originalPointIdx), infIdx);

    setPredWrapperConstant(dPredWrapper);

    return {v0, v1, v2};
}

void GpuDel::splitAndFlip()
{
    int insLoop = 0;
    while (availPtNum > 0)
    {
        splitTri();
        if (doFlipping)
        {
            flipLoop(CircleFastOrientFast);
        }
        ++insLoop;
    }
    if (!doFlipping)
    {
        flipLoop(CircleFastOrientFast);
    }

    markSpecialTris();
    flipLoop(CircleExactOrientSoS);

    // Insert constraints if needed
    if (constraintVec.size() > 0)
    {
        insertConstraints();
    }

    flipLoop(CircleFastOrientFast);
    markSpecialTris();
    flipLoop(CircleExactOrientSoS);

    if (inputPtr->isProfiling(ProfDiag))
    {
        std::cout << "\nInsert loops: " << insLoop << std::endl;
        std::cout << "Compact: " << std::endl;
        diagLogCompact.printCount();
        std::cout << "Collect: " << std::endl;
        diagLogCollect.printCount();
    }
}

void GpuDel::splitTri()
{
    startTiming(ProfDefault);

    IntDVec triToVert   = memPool.allocateAny<int>(triMaxNum);
    IntDVec splitTriVec = memPool.allocateAny<int>(pointNum);
    IntDVec insTriMap   = memPool.allocateAny<int>(triMaxNum);

    auto triNum   = static_cast<int>(triVec.size());
    int  noSample = pointNum / triNum > MaxSamplePerTri ? triNum * MaxSamplePerTri : pointNum;
    getRankedPoints(triNum, noSample, triToVert);
    insertTriNum = thrust_copyIf_TriHasVert(triToVert, splitTriVec);

    const int splitTriNum = triNum + DIM * insertTriNum;
    if (inputPtr->isProfiling(ProfDiag))
    {
        std::cout << "Insert: " << insertTriNum << " Tri from: " << triNum << " to: " << splitTriNum << std::endl;
    }
    shiftTriIfNeed(triNum, triToVert, splitTriVec);
    makeTriMap(splitTriNum, triNum, splitTriVec, insTriMap);
    stopTiming(ProfDefault, outputPtr->stats.splitTime);

    splitPoints(triNum, triToVert, insTriMap);
    splitOldTriIntoNew(triNum, triToVert, splitTriVec, insTriMap);

    memPool.release(triToVert);
    memPool.release(splitTriVec);
    memPool.release(insTriMap);
    availPtNum -= insertTriNum;
}

void GpuDel::getRankedPoints(int triNum, int noSample, IntDVec &triToVert)
{
    IntDVec triCircleVec = memPool.allocateAny<int>(triMaxNum);
    triCircleVec.assign(triNum, INT_MIN);
    IntDVec vertCircleVec = memPool.allocateAny<int>(pointNum);
    vertCircleVec.resize(noSample);
    kerVoteForPoint<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(vertexTriVec),
                                                        toKernelPtr(triVec),
                                                        toKernelPtr(vertCircleVec),
                                                        toKernelPtr(triCircleVec),
                                                        noSample);
    CudaCheckError();
    triToVert.assign(triNum, INT_MAX);
    kerPickWinnerPoint<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(vertexTriVec),
                                                           toKernelPtr(vertCircleVec),
                                                           toKernelPtr(triCircleVec),
                                                           toKernelPtr(triToVert),
                                                           noSample);
    CudaCheckError();
    memPool.release(vertCircleVec);
    memPool.release(triCircleVec);
}

void GpuDel::shiftTriIfNeed(int &triNum, IntDVec &triToVert, IntDVec &splitTriVec)
{
    if (availPtNum - insertTriNum < insertTriNum && insertTriNum < 0.1 * pointNum)
    {
        doFlipping = false; // Do not flip if there's just a few points
    }
    if (!inputPtr->noReorder && doFlipping)
    {
        stopTiming(ProfDefault, outputPtr->stats.splitTime);
        shiftTri(triToVert, splitTriVec);
        triNum = -1; // Mark that we have shifted the array
        startTiming(ProfDefault);
    }
}

void GpuDel::shiftTri(IntDVec &triToVert, IntDVec &splitTriVec)
{
    startTiming(ProfDefault);

    const auto triNum   = static_cast<int>(triVec.size() + 2 * splitTriVec.size());
    IntDVec    shiftVec = memPool.allocateAny<int>(triMaxNum);
    thrust_scan_TriHasVert(triToVert, shiftVec);
    shiftExpandVec(shiftVec, triVec, triNum);
    shiftExpandVec(shiftVec, triInfoVec, triNum);
    shiftExpandVec(shiftVec, triToVert, triNum);
    shiftOppVec(shiftVec, oppVec, triNum);

    kerShiftTriIdx<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(vertexTriVec), toKernelPtr(shiftVec));
    CudaCheckError();
    kerShiftTriIdx<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(splitTriVec), toKernelPtr(shiftVec));
    CudaCheckError();
    memPool.release(shiftVec);
    stopTiming(ProfDefault, outputPtr->stats.sortTime);
}

template <typename T>
void GpuDel::shiftExpandVec(IntDVec &shiftVec, DevVector<T> &dataVec, int size)
{
    DevVector<T> tempVec = memPool.allocateAny<T>(size);
    tempVec.resize(size);
    kerShift<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(shiftVec), toKernelPtr(dataVec), toKernelPtr(tempVec));
    CudaCheckError();
    dataVec.copyFrom(tempVec);
    memPool.release(tempVec);
}

void GpuDel::shiftOppVec(IntDVec &shiftVec, TriOppDVec &dataVec, int size)
{
    TriOppDVec tempVec = memPool.allocateAny<TriOpp>(size);
    tempVec.resize(size);
    kerShiftOpp<<<BlocksPerGrid, ThreadsPerBlock>>>(
        toKernelArray(shiftVec), toKernelPtr(dataVec), toKernelPtr(tempVec), size);
    CudaCheckError();
    dataVec.copyFrom(tempVec);
    memPool.release(tempVec);
}

void GpuDel::makeTriMap(int splitTriNum, int triNum, const IntDVec &splitTriVec, IntDVec &insTriMap)
{
    insTriMap.assign((triNum < 0) ? splitTriNum : triNum, -1);
    thrust_scatterSequenceMap(splitTriVec, insTriMap);
    expandTri(splitTriNum);
}

void GpuDel::expandTri(int newTriNum)
{
    triVec.expand(newTriNum);
    oppVec.expand(newTriNum);
    triInfoVec.expand(newTriNum);
}

void GpuDel::splitPoints(int triNum, IntDVec &triToVert, IntDVec &insTriMap)
{
    startTiming(ProfDefault);
    IntDVec exactCheckVec = memPool.allocateAny<int>(pointNum);
    counters.renew();
    kerSplitPointsFast<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(vertexTriVec),
                                                           toKernelPtr(triToVert),
                                                           toKernelPtr(triVec),
                                                           toKernelPtr(insTriMap),
                                                           toKernelPtr(exactCheckVec),
                                                           counters.ptr(),
                                                           triNum,
                                                           insertTriNum);
    kerSplitPointsExactSoS<<<PredBlocksPerGrid, PredThreadsPerBlock>>>(toKernelPtr(vertexTriVec),
                                                                       toKernelPtr(triToVert),
                                                                       toKernelPtr(triVec),
                                                                       toKernelPtr(insTriMap),
                                                                       toKernelPtr(exactCheckVec),
                                                                       counters.ptr(),
                                                                       triNum,
                                                                       insertTriNum);
    CudaCheckError();
    memPool.release(exactCheckVec);
    stopTiming(ProfDefault, outputPtr->stats.relocateTime);
}

void GpuDel::splitOldTriIntoNew(int triNum, IntDVec &triToVert, IntDVec &splitTriVec, IntDVec &insTriMap)
{
    startTiming(ProfDefault);
    kerSplitTri<<<BlocksPerGrid, 32>>>(toKernelArray(splitTriVec),
                                       toKernelPtr(triVec),
                                       toKernelPtr(oppVec),
                                       toKernelPtr(triInfoVec),
                                       toKernelPtr(insTriMap),
                                       toKernelPtr(triToVert),
                                       triNum);
    CudaCheckError();
    stopTiming(ProfDefault, outputPtr->stats.splitTime);
}

void GpuDel::flipLoop(CheckDelaunayMode checkMode)
{
    startTiming(ProfDefault);

    flipVec   = memPool.allocateAny<FlipItem>(triMaxNum);
    triMsgVec = memPool.allocateAny<int2>(triMaxNum);
    actTriVec = memPool.allocateAny<int>(triMaxNum);

    triMsgVec.assign(triMaxNum, make_int2(-1, -1));

    int flipLoop = 0;
    actTriMode   = ActTriMarkCompact;
    diagLogPtr   = &diagLogCompact;
    while (flip(checkMode))
    {
        ++flipLoop;
    }

    stopTiming(ProfDefault, outputPtr->stats.flipTime);
    relocateAll();
    memPool.release(triMsgVec);
    memPool.release(flipVec);
    memPool.release(actTriVec);
}

bool GpuDel::flip(CheckDelaunayMode checkMode)
{
    startTiming(ProfDetail);

    ++diagLogPtr->_flipLoop;
    const auto triNum = static_cast<int>(triVec.size());

    compactActiveTriangles();

    auto orgActNum = static_cast<int>(actTriVec.size());
    if (inputPtr->isProfiling(ProfDiag))
    {
        numActiveVec.push_back(orgActNum);
        if (orgActNum == 0 ||
            (checkMode != CircleExactOrientSoS && orgActNum < PredBlocksPerGrid * PredThreadsPerBlock))
        {
            numFlipVec.push_back(0);
            timeCheckVec.push_back(0.0);
            timeFlipVec.push_back(0.0);
            numCircleVec.push_back(0);
        }
    }
    restartTiming(ProfDetail, diagLogPtr->_t[0]);

    // No more work
    if (0 == orgActNum)
    {
        return false;
    }
    // Little work, leave it for the Exact iterations
    if (checkMode != CircleExactOrientSoS && orgActNum < PredBlocksPerGrid * PredThreadsPerBlock)
    {
        return false;
    }
    selectMode(triNum, orgActNum);

    IntDVec   flipToTri = memPool.allocateAny<int>(triMaxNum);
    const int flipNum   = getFlipNum(checkMode, triNum, orgActNum, flipToTri);

    restartTiming(ProfDetail, diagLogPtr->_t[3]);

    // Preparation for the actual flipping. Include several steps
    if (inputPtr->isProfiling(ProfDiag))
    {
        const int circleNum = thrust_sum(circleCountVec);
        diagLogPtr->_circleCount += circleNum;
        const int rejFlipNum = thrust_sum(rejFlipVec);
        diagLogPtr->_rejFlipCount += rejFlipNum;
        diagLogPtr->_totFlipNum += flipNum;
        std::cout << "Acts: " << orgActNum << " Flips: " << flipNum << " ( " << rejFlipNum << " )"
                  << " circle: " << circleNum
                  << " Exact: " << (checkMode == CircleExactOrientSoS ? counters[CounterExact] : -1) << std::endl;
        numCircleVec.push_back(circleNum);
        startTiming(ProfDetail);
    }

    if (0 == flipNum)
    {
        numCircleVec.push_back(0);
        timeFlipVec.push_back(0);
        memPool.release(flipToTri);
        return false;
    }

    originalFlipNum.push_back(getOriginalFlipNumAndExpandFlipVec(flipNum));
    doFlippingAndUpdateOppTri(orgActNum, flipNum, flipToTri);

    memPool.release(flipToTri);
    return true;
}

void GpuDel::compactActiveTriangles()
{
    switch (actTriMode)
    {
    case ActTriMarkCompact:
        thrust_copyIf_IsActiveTri(triInfoVec, actTriVec);
        break;
    case ActTriCollectCompact:
        IntDVec temp = memPool.allocateAny<int>(triMaxNum, true);
        compactIfNegative(actTriVec, temp);
        break;
    }
}

void GpuDel::selectMode(const int triNum, int orgActNum)
{ // See if there's little work enough to switch to collect mode.
    // Safety check: make sure there's enough space to collect
    if (orgActNum < BlocksPerGrid * ThreadsPerBlock && orgActNum * 2 < actTriVec.capacity() && orgActNum * 2 < triNum)
    {
        actTriMode = ActTriCollectCompact;
        diagLogPtr = &diagLogCollect;
    }
    else
    {
        actTriMode = ActTriMarkCompact;
        diagLogPtr = &diagLogCompact;
    }
}

int GpuDel::getFlipNum(CheckDelaunayMode checkMode, int triNum, int orgActNum, IntDVec &flipToTri)
{
    IntDVec triVoteVec = memPool.allocateAny<int>(triMaxNum);
    getTriVotes(checkMode, triNum, orgActNum, triVoteVec);
    double prevTime = diagLogPtr->_t[1];
    restartTiming(ProfDetail, diagLogPtr->_t[1]);
    flipToTri.resize(orgActNum);
    kerMarkRejectedFlips<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelPtr(actTriVec),
                                                             toKernelPtr(oppVec),
                                                             toKernelPtr(triVoteVec),
                                                             toKernelPtr(triInfoVec),
                                                             toKernelPtr(flipToTri),
                                                             orgActNum,
                                                             inputPtr->isProfiling(ProfDiag) ? toKernelPtr(rejFlipVec)
                                                                                             : nullptr);
    CudaCheckError();
    memPool.release(triVoteVec);

    restartTiming(ProfDetail, diagLogPtr->_t[2]);
    IntDVec   temp    = memPool.allocateAny<int>(triMaxNum, true);
    const int flipNum = compactIfNegative(flipToTri, temp);
    if (inputPtr->isProfiling(ProfDiag))
    {
        numFlipVec.push_back(flipNum);
        timeCheckVec.push_back(diagLogPtr->_t[1] - prevTime);
    }
    return flipNum;
}

void GpuDel::getTriVotes(CheckDelaunayMode checkMode, int triNum, int orgActNum, IntDVec &triVoteVec)
{
    if (inputPtr->isProfiling(ProfDiag))
    {
        circleCountVec.assign(triNum, 0);
        rejFlipVec.assign(triNum, 0);
    }
    triVoteVec.assign(triNum, INT_MAX);
    dispatchCheckDelaunay(checkMode, orgActNum, triVoteVec);
}

void GpuDel::dispatchCheckDelaunay(CheckDelaunayMode checkMode, int orgActNum, IntDVec &triVoteVec)
{
    switch (checkMode)
    {
    case CircleFastOrientFast:
        kerCheckDelaunayFast<<<BlocksPerGrid, ThreadsPerBlock>>>(
            toKernelPtr(actTriVec),
            toKernelPtr(triVec),
            toKernelPtr(oppVec),
            toKernelPtr(triInfoVec),
            toKernelPtr(triVoteVec),
            orgActNum,
            inputPtr->isProfiling(ProfDiag) ? toKernelPtr(circleCountVec) : nullptr);
        CudaCheckError();
        break;
    case CircleExactOrientSoS:
        // Reuse this array to save memory
        Int2DVec &exactCheckVi = triMsgVec;
        counters.renew();
        kerCheckDelaunayExact_Fast<<<BlocksPerGrid, ThreadsPerBlock>>>(
            toKernelPtr(actTriVec),
            toKernelPtr(triVec),
            toKernelPtr(oppVec),
            toKernelPtr(triInfoVec),
            toKernelPtr(triVoteVec),
            toKernelPtr(exactCheckVi),
            orgActNum,
            counters.ptr(),
            inputPtr->isProfiling(ProfDiag) ? toKernelPtr(circleCountVec) : nullptr);
        kerCheckDelaunayExact_Exact<<<PredBlocksPerGrid, PredThreadsPerBlock>>>(
            toKernelPtr(triVec),
            toKernelPtr(oppVec),
            toKernelPtr(triVoteVec),
            toKernelPtr(exactCheckVi),
            counters.ptr(),
            inputPtr->isProfiling(ProfDiag) ? toKernelPtr(circleCountVec) : nullptr);
        CudaCheckError();
        break;
    }
}

int GpuDel::getOriginalFlipNumAndExpandFlipVec(const int flipNum)
{
    auto orgFlipNum = static_cast<int>(flipVec.size());
    int  expFlipNum = orgFlipNum + flipNum;

    if (expFlipNum > flipVec.capacity())
    {
        stopTiming(ProfDetail, diagLogPtr->_t[4]);
        stopTiming(ProfDefault, outputPtr->stats.flipTime);
        relocateAll();
        startTiming(ProfDefault);
        startTiming(ProfDetail);

        orgFlipNum = 0;
        expFlipNum = flipNum;
    }

    flipVec.grow(expFlipNum);
    return orgFlipNum;
}

void GpuDel::doFlippingAndUpdateOppTri(int orgActNum, int flipNum, IntDVec &flipToTri)
{
    // triMsgVec contains two components.
    // - .x is the encoded new neighbor information
    // - .y is the flipIdx as in the flipVec (d_i.e. globIdx)
    // As such, we do not need to initialize it to -1 to
    // know which tris are not flipped in the current rount.
    // We can rely on the flipIdx being > or < than orgFlipIdx.
    // Note that we have to initialize everything to -1
    // when we clear the flipVec and reset the flip indexing.
    //
    triMsgVec.resize(triVec.size());
    if (actTriMode == ActTriCollectCompact)
    {
        actTriVec.grow(orgActNum + flipNum);
    }
    restartTiming(ProfDetail, diagLogPtr->_t[4]);

    // Flipping, 32 ThreadsPerBlock is optimal
    kerFlip<<<BlocksPerGrid, 32>>>(toKernelArray(flipToTri),
                                   toKernelPtr(triVec),
                                   toKernelPtr(oppVec),
                                   toKernelPtr(triInfoVec),
                                   toKernelPtr(triMsgVec),
                                   (actTriMode == ActTriCollectCompact) ? toKernelPtr(actTriVec) : nullptr,
                                   toKernelPtr(flipVec),
                                   nullptr,
                                   nullptr,
                                   originalFlipNum.back(),
                                   orgActNum);
    CudaCheckError();

    // Update oppTri
    kerUpdateOpp<<<BlocksPerGrid, 32>>>(toKernelPtr(flipVec) + originalFlipNum.back(),
                                        toKernelPtr(oppVec),
                                        toKernelPtr(triMsgVec),
                                        toKernelPtr(flipToTri),
                                        originalFlipNum.back(),
                                        flipNum);
    CudaCheckError();
    double prevTime = diagLogPtr->_t[5];
    stopTiming(ProfDetail, diagLogPtr->_t[5]);
    if (inputPtr->isProfiling(ProfDiag))
    {
        timeFlipVec.push_back(diagLogPtr->_t[5] - prevTime);
    }
}

void GpuDel::relocateAll()
{
    if (flipVec.size() == 0)
    {
        return;
    }

    startTiming(ProfDefault);

    if (availPtNum > 0)
    {
        IntDVec triToFlip = memPool.allocateAny<int>(triMaxNum);
        triToFlip.assign(triVec.size(), -1);
        rebuildTriPtrAfterFlipping(triToFlip);
        relocatePoints(triToFlip);
        memPool.release(triToFlip);
    }

    // Just clean up the flips
    flipVec.resize(0);
    originalFlipNum.clear();

    // Reset the triMsgVec
    triMsgVec.assign(triMaxNum, make_int2(-1, -1));
    stopTiming(ProfDefault, outputPtr->stats.relocateTime);
}

void GpuDel::rebuildTriPtrAfterFlipping(IntDVec &triToFlip)
{
    auto nextFlipNum = static_cast<int>(flipVec.size());
    for (int i = static_cast<int>(originalFlipNum.size()) - 1; i >= 0; --i)
    {
        int prevFlipNum = originalFlipNum[i];
        int flipNum     = nextFlipNum - prevFlipNum;

        kerUpdateFlipTrace<<<BlocksPerGrid, ThreadsPerBlock>>>(
            toKernelPtr(flipVec), toKernelPtr(triToFlip), prevFlipNum, flipNum);

        nextFlipNum = prevFlipNum;
    }
    CudaCheckError();
}

void GpuDel::relocatePoints(IntDVec &triToFlip)
{
    IntDVec exactCheckVec = memPool.allocateAny<int>(pointNum);
    counters.renew();
    kerRelocatePointsFast<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(vertexTriVec),
                                                              toKernelPtr(triToFlip),
                                                              toKernelPtr(flipVec),
                                                              toKernelPtr(exactCheckVec),
                                                              counters.ptr());

    kerRelocatePointsExact<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelPtr(vertexTriVec),
                                                               toKernelPtr(triToFlip),
                                                               toKernelPtr(flipVec),
                                                               toKernelPtr(exactCheckVec),
                                                               counters.ptr());
    CudaCheckError();
    memPool.release(exactCheckVec);
}

void GpuDel::markSpecialTris()
{
    startTiming(ProfDetail);
    kerMarkSpecialTris<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(triInfoVec), toKernelPtr(oppVec));
    CudaCheckError();
    stopTiming(ProfDetail, diagLogPtr->_t[0]);
}

void GpuDel::insertConstraints()
{
    startTiming(ProfDefault);

    initForConstraintInsertion();
    triConsVec = memPool.allocateAny<int>(triVec.size());
    flipVec    = memPool.allocateAny<FlipItem>(triMaxNum);
    triMsgVec  = memPool.allocateAny<int2>(triMaxNum);
    actTriVec  = memPool.allocateAny<int>(triMaxNum);

    triConsVec.assign(triVec.size(), -1);
    triMsgVec.assign(triMaxNum, make_int2(-1, -1));

    int outerLoop  = 0;
    int flipLoop   = 0;
    int totFlipNum = 0;
    int flipNum;

    while (markIntersections())
    {
        if (inputPtr->isProfiling(ProfDiag))
        {
            std::cout << "Iter " << (outerLoop + 1) << std::endl;
        }
        // Collect active triangles
        thrust_copyIf_IsNotNegative(triConsVec, actTriVec);
        int innerLoop = 0;
        while (flipConstraints(flipNum))
        {
            totFlipNum += flipNum;
            ++flipLoop;
            ++innerLoop;
            if (innerLoop == 5)
            {
                break;
            }
        }
        ++outerLoop;
        // Mark all the possibly modified triangles as Alive + Changed (3).
        thrust_scatterConstantMap(actTriVec, triInfoVec, 3);
    }

    if (inputPtr->isProfiling(ProfDiag))
    {
        std::cout << "ConsFlip: Outer loop = " << outerLoop << ", inner loop = " << flipLoop
                  << ", total flip = " << totFlipNum << std::endl;
    }

    memPool.release(triConsVec);
    memPool.release(triMsgVec);
    memPool.release(actTriVec);
    memPool.release(flipVec);
    stopTiming(ProfDefault, outputPtr->stats.constraintTime);
}

void GpuDel::initForConstraintInsertion()
{
    if (!inputPtr->noSort)
    {
        // Update vertex indices of constraints
        IntDVec mapVec = memPool.allocateAny<int>(pointNum);
        mapVec.resize(pointNum);
        thrust_scatterSequenceMap(originalPointIdx, mapVec);
        thrust::device_ptr<int> segInt((int *)toKernelPtr(constraintVec));
        thrust::gather(segInt, segInt + static_cast<long>(constraintVec.size()) * 2, mapVec.begin(), segInt);
        memPool.release(mapVec);
    }

    // Construct
    vertexTriVec.resize(pointNum);
    kerMapTriToVert<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(triVec), toKernelPtr(vertexTriVec));
    CudaCheckError();
    // Initialize list of active constraints
    thrust::sequence(actConsVec.begin(), actConsVec.end());
}

bool GpuDel::markIntersections()
{
    counters.renew();
    kerMarkTriConsIntersectionFast<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(actConsVec),
                                                                       toKernelPtr(constraintVec),
                                                                       toKernelPtr(triVec),
                                                                       toKernelPtr(oppVec),
                                                                       toKernelPtr(triInfoVec),
                                                                       toKernelPtr(vertexTriVec),
                                                                       toKernelPtr(triConsVec),
                                                                       counters.ptr());
    CudaCheckError();
    kerMarkTriConsIntersectionExact<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(actConsVec),
                                                                        toKernelPtr(constraintVec),
                                                                        toKernelPtr(triVec),
                                                                        toKernelPtr(oppVec),
                                                                        toKernelPtr(triInfoVec),
                                                                        toKernelPtr(vertexTriVec),
                                                                        toKernelPtr(triConsVec),
                                                                        counters.ptr());
    CudaCheckError();
    return (counters[CounterFlag] == 1);
}

bool GpuDel::flipConstraints(int &flipNum)
{
    const auto triNum = triVec.size();
    const auto actNum = actTriVec.size();

    // Vote for flips
    if (inputPtr->isProfiling(ProfDiag))
    {
        rejFlipVec.assign(triNum, 0);
    }
    updatePairStatus();

    IntDVec triVoteVec = memPool.allocateAny<int>(triMaxNum);
    triVoteVec.assign(triNum, INT_MAX);
    checkConsFlipping(triVoteVec);
    IntDVec flipToTri = memPool.allocateAny<int>(triMaxNum);
    flipToTri.resize(actNum);
    updateFlipConsNum(flipNum, triVoteVec, flipToTri);
    memPool.release(triVoteVec);

    if (0 == flipNum)
    {
        memPool.release(flipToTri);
        return false;
    }

    auto orgFlipNum = static_cast<int>(flipVec.size());
    int  expFlipNum = orgFlipNum + flipNum;

    if (expFlipNum > flipVec.capacity())
    {
        flipVec.resize(0);
        triMsgVec.assign(triMaxNum, make_int2(-1, -1));
        orgFlipNum = 0;
        expFlipNum = flipNum;
    }

    doFlippingAndUpdateOppTri(flipNum, orgFlipNum, expFlipNum, flipToTri);
    memPool.release(flipToTri);
    return true;
}

void GpuDel::doFlippingAndUpdateOppTri(int flipNum, int orgFlipNum, int expFlipNum, IntDVec &flipToTri)
{
    triMsgVec.resize(triVec.size());
    flipVec.grow(expFlipNum);
    if (inputPtr->isProfiling(ProfDiag))
    {
        const int rejFlipNum = thrust_sum(rejFlipVec);
        std::cout << "  ConsFlips: " << flipNum << " ( " << rejFlipNum << " )" << std::endl;
    }

    kerFlip<<<BlocksPerGrid, 32>>>(toKernelArray(flipToTri),
                                   toKernelPtr(triVec),
                                   toKernelPtr(oppVec),
                                   nullptr,
                                   toKernelPtr(triMsgVec),
                                   nullptr,
                                   toKernelPtr(flipVec),
                                   toKernelPtr(triConsVec),
                                   toKernelPtr(vertexTriVec),
                                   orgFlipNum,
                                   0);
    CudaCheckError();
    kerUpdateOpp<<<BlocksPerGrid, 32>>>(toKernelPtr(flipVec) + orgFlipNum,
                                        toKernelPtr(oppVec),
                                        toKernelPtr(triMsgVec),
                                        toKernelPtr(flipToTri),
                                        orgFlipNum,
                                        flipNum);
    CudaCheckError();
}

void GpuDel::updatePairStatus()
{
    IntDVec exactVec = memPool.allocateAny<int>(triMaxNum);
    counters.renew();
    kerUpdatePairStatusFast<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(actTriVec),
                                                                toKernelPtr(triConsVec),
                                                                toKernelPtr(triVec),
                                                                toKernelPtr(oppVec),
                                                                toKernelPtr(triInfoVec),
                                                                toKernelPtr(exactVec),
                                                                counters.ptr());
    CudaCheckError();
    kerUpdatePairStatusExact<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(actTriVec),
                                                                 toKernelPtr(triConsVec),
                                                                 toKernelPtr(triVec),
                                                                 toKernelPtr(oppVec),
                                                                 toKernelPtr(triInfoVec),
                                                                 toKernelPtr(exactVec),
                                                                 counters.ptr());
    CudaCheckError();
    memPool.release(exactVec);
}

void GpuDel::checkConsFlipping(IntDVec &triVoteVec)
{
    IntDVec exactVec = memPool.allocateAny<int>(triMaxNum);
    counters.renew();
    kerCheckConsFlippingFast<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelArray(actTriVec),
                                                                 toKernelPtr(triConsVec),
                                                                 toKernelPtr(triInfoVec),
                                                                 toKernelPtr(triVec),
                                                                 toKernelPtr(oppVec),
                                                                 toKernelPtr(triVoteVec),
                                                                 toKernelPtr(exactVec),
                                                                 counters.ptr());
    CudaCheckError();
    kerCheckConsFlippingExact<<<BlocksPerGrid, ThreadsPerBlock>>>(toKernelPtr(triConsVec),
                                                                  toKernelPtr(triInfoVec),
                                                                  toKernelPtr(triVec),
                                                                  toKernelPtr(oppVec),
                                                                  toKernelPtr(triVoteVec),
                                                                  toKernelPtr(exactVec),
                                                                  counters.ptr());
    CudaCheckError();
    memPool.release(exactVec);
}

void GpuDel::updateFlipConsNum(int &flipNum, IntDVec &triVoteVec, IntDVec &flipToTri)
{
    kerMarkRejectedConsFlips<<<BlocksPerGrid, ThreadsPerBlock>>>(
        toKernelArray(actTriVec),
        toKernelPtr(triConsVec),
        toKernelPtr(triVoteVec),
        toKernelPtr(triInfoVec),
        toKernelPtr(oppVec),
        toKernelPtr(flipToTri),
        inputPtr->isProfiling(ProfDiag) ? toKernelPtr(rejFlipVec) : nullptr);
    CudaCheckError();

    IntDVec temp = memPool.allocateAny<int>(triMaxNum, true);
    flipNum      = compactIfNegative(flipToTri, temp);
}

void GpuDel::outputToHost()
{
    startTiming(ProfDefault);

    kerMarkInfinityTri<<<BlocksPerGrid, ThreadsPerBlock>>>(
        toKernelArray(triVec), toKernelPtr(triInfoVec), toKernelPtr(oppVec), infIdx);
    CudaCheckError();
    compactTris();

    if (!inputPtr->noSort)
    {
        // Change the indices back to the original order
        kerUpdateVertIdx<<<BlocksPerGrid, ThreadsPerBlock>>>(
            toKernelArray(triVec), toKernelPtr(triInfoVec), toKernelPtr(originalPointIdx));
        CudaCheckError();
    }

    // Copy to host
    triVec.copyToHost(outputPtr->triVec);
    oppVec.copyToHost(outputPtr->triOppVec);

    stopTiming(ProfDefault, outputPtr->stats.outTime);
    std::cout << "# Triangles:     " << triVec.size() << std::endl;
}

void GpuDel::compactTris()
{
    const auto triNum = static_cast<int>(triVec.size());

    IntDVec prefixVec = memPool.allocateAny<int>(triMaxNum);
    prefixVec.resize(triNum);
    thrust_scan_TriAliveStencil(triInfoVec, prefixVec);

    int newTriNum = prefixVec[triNum - 1];
    int freeNum   = triNum - newTriNum;
    IntDVec freeVec = memPool.allocateAny<int>(triMaxNum);
    freeVec.resize(freeNum);

    kerCollectFreeSlots<<<BlocksPerGrid, ThreadsPerBlock>>>(
        toKernelPtr(triInfoVec), toKernelPtr(prefixVec), toKernelPtr(freeVec), newTriNum);
    CudaCheckError();
    // Make map
    kerMakeCompactMap<<<BlocksPerGrid, ThreadsPerBlock>>>(
        toKernelArray(triInfoVec), toKernelPtr(prefixVec), toKernelPtr(freeVec), newTriNum);
    CudaCheckError();
    // Reorder the tets
    kerCompactTris<<<BlocksPerGrid, ThreadsPerBlock>>>(
        toKernelArray(triInfoVec), toKernelPtr(prefixVec), toKernelPtr(triVec), toKernelPtr(oppVec), newTriNum);
    CudaCheckError();

    triInfoVec.resize(newTriNum);
    triVec.resize(newTriNum);
    oppVec.resize(newTriNum);
    memPool.release(freeVec);
    memPool.release(prefixVec);
}

void GpuDel::cleanup()
{
    thrust_free_all();

    memPool.free();

    pointVec.free();
    constraintVec.free();
    triVec.free();
    oppVec.free();
    triInfoVec.free();
    originalPointIdx.free();
    vertexTriVec.free();
    counters.free();
    actConsVec.free();

    originalFlipNum.clear();

    dPredWrapper.cleanup();

    circleCountVec.free();
    rejFlipVec.free();

    outputPtr->stats.reset();
    diagLogCompact.reset();
    diagLogCollect.reset();

    numActiveVec.clear();
    numFlipVec.clear();
    numCircleVec.clear();
    timeCheckVec.clear();
    timeFlipVec.clear();
}

void GpuDel::startTiming(ProfLevel level)
{
    if (inputPtr->isProfiling(level))
    {
        profTimer[level].start();
    }
}

void GpuDel::pauseTiming(ProfLevel level)
{
    if (inputPtr->isProfiling(level))
    {
        profTimer[level].pause();
    }
}

void GpuDel::stopTiming(ProfLevel level, double &accuTime)
{
    if (inputPtr->isProfiling(level))
    {
        profTimer[level].stop();
        accuTime += profTimer[level].value();
    }
}

void GpuDel::restartTiming(ProfLevel level, double &accuTime)
{
    stopTiming(level, accuTime);
    startTiming(level);
}