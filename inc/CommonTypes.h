#ifndef DELAUNAY_GENERATOR_COMMONTYPES_H
#define DELAUNAY_GENERATOR_COMMONTYPES_H

// STL
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include "GPU/CudaWrapper.h"
#include "GPU/MemoryManager.h"

#define DIM 2
#define DEG (DIM + 1)

typedef unsigned char uchar;

#define INLINE_H_D __forceinline__ __host__ __device__

enum Counter : int
{
    CounterExact,
    CounterFlag,
    CounterNum
};

enum Orient
{
    OrientNeg  = -1,
    OrientZero = +0,
    OrientPos  = +1
};

INLINE_H_D
Orient flipOrient(Orient ord)
{
#ifdef __CUDA_ARCH__
    CudaAssert(OrientZero != ord);
#else
    assert(OrientZero != ord);
#endif
    return (OrientPos == ord) ? OrientNeg : OrientPos;
}

// Our 2D orientation is the same as Shewchuk
INLINE_H_D
Orient ortToOrient(double det)
{
    return (det > 0) ? OrientPos : ((det < 0) ? OrientNeg : OrientZero);
}

enum Side
{
    SideIn   = -1,
    SideZero = +0,
    SideOut  = +1
};

INLINE_H_D Side cicToSide(double det)
{
    return (det < 0) ? SideOut : ((det > 0) ? SideIn : SideZero);
}

INLINE_H_D void setTriIdxVi(int &output, int oldVi, int ni, int newVi)
{
    output -= (0xF) << (oldVi * 4);
    output += ((ni << 2) + newVi) << (oldVi * 4);
}

INLINE_H_D bool almost_zero(double x)
{
    return std::abs(x) < 1e-9;
}

INLINE_H_D bool almost_equal(double x, double y){
    return std::abs(x - y) < 1e-9;
}

struct Point
{
    double _p[3] = {0, 0, 0};

    Point()=default;

    INLINE_H_D
    Point(double x, double y, double z) : _p{x, y, z} {};

    INLINE_H_D
    bool operator<(const Point &pt) const
    {
        if (_p[0] < pt._p[0])
            return true;
        if (_p[0] > pt._p[0])
            return false;
        if (_p[1] < pt._p[1])
            return true;

        return false;
    }

    INLINE_H_D
    bool operator==(const Point &pt) const
    {
        return almost_equal(_p[0], pt._p[0]) && almost_equal(_p[1], pt._p[1]);
    }

    INLINE_H_D
    bool operator!=(const Point &pt) const
    {
        return !(*this==pt);
    }
};

struct Tri
{
    int _v[3] = {0, 0, 0};

    Tri() = default;

    INLINE_H_D
    Tri(int v0, int v1, int v2) : _v{v0, v1, v2} {};

    INLINE_H_D bool has(int v) const
    {
        return (_v[0] == v || _v[1] == v || _v[2] == v);
    }

    INLINE_H_D int getIndexOf(int v) const
    {
        if (_v[0] == v)
            return 0;
        if (_v[1] == v)
            return 1;
        if (_v[2] == v)
            return 2;
#ifdef __CUDA_ARCH__
        CudaAssert(false);
#else
        assert(false);
#endif
    }

    INLINE_H_D bool operator==(const Tri &tri) const
    {
        return _v[0] == tri._v[0] && _v[1] == tri._v[1] && _v[2] == tri._v[2];
    }

    INLINE_H_D bool operator!=(const Tri &tri) const
    {
        return !(*this == tri);
    }
};

///////////////////////////////////////////////////////////////////// TriOpp //

// ...76543210
//        ^^^^ vi (2 bits)
//        ||__ constraint
//        |___ special
// Rest is triIdx

template <typename T>
INLINE_H_D bool isBitSet(T c, int bit)
{
    return (1 == ((c >> bit) & 1));
}

template <typename T>
INLINE_H_D void setBitState(T &c, int bit, bool state)
{
    const T val = (1 << bit);
    c           = state ? (c | val) : (c & ~val);
}

// Get the opp tri and vi
// Retain some states: constraint
// Clear some states:  special
INLINE_H_D int getOppValTriVi(int val)
{
    return (val & ~0x08);
}

INLINE_H_D int getOppValTri(int val)
{
    return (val >> 4);
}

INLINE_H_D int getOppValVi(int val)
{
    return (val & 3);
}

INLINE_H_D bool isOppValConstraint(int val)
{
    return isBitSet(val, 2);
}

INLINE_H_D void setOppValTri(int &val, int idx)
{
    val = (val & 0x0F) | (idx << 4);
}

// Set the opp tri and vi
// Retain some states: constraint
// Clear some states:  special
INLINE_H_D void setOppValTriVi(int &val, int idx, int vi)
{
    val = (idx << 4) | (val & 0x04) | vi;
}

INLINE_H_D int getTriIdx(int input, int oldVi)
{
    int idxVi = (input >> (oldVi * 4)) & 0xf;
    return (idxVi >> 2) & 0x3;
}

INLINE_H_D int getTriVi(int input, int oldVi)
{
    int idxVi = (input >> (oldVi * 4)) & 0xf;
    return idxVi & 0x3;
}

struct TriOpp
{
    int _t[3] = {0, 0, 0};

    INLINE_H_D void setOpp(int vi, int triIdx, int oppTriVi)
    {
        _t[vi] = (triIdx << 4) | oppTriVi;
    }

    INLINE_H_D void setOpp(int vi, int triIdx, int oppTriVi, bool isConstraint)
    {
        _t[vi] = (triIdx << 4) | (isConstraint << 2) | oppTriVi;
    }

    INLINE_H_D void setOppTriVi(int vi, int triIdx, int oppTriVi)
    {
        setOppValTriVi(_t[vi], triIdx, oppTriVi);
    }

    INLINE_H_D void setOppConstraint(int vi, bool val)
    {
        setBitState(_t[vi], 2, val);
    }

    INLINE_H_D void setOppSpecial(int vi, bool state)
    {
        setBitState(_t[vi], 3, state);
    }

    INLINE_H_D bool isNeighbor(int triIdx) const
    {
        return ((_t[0] >> 4) == triIdx || (_t[1] >> 4) == triIdx || (_t[2] >> 4) == triIdx);
    }

    INLINE_H_D int getIdxOf(int triIdx) const
    {
        if ((_t[0] >> 4) == triIdx)
            return 0;
        if ((_t[1] >> 4) == triIdx)
            return 1;
        if ((_t[2] >> 4) == triIdx)
            return 2;
        return -1;
    }

    INLINE_H_D bool isOppSpecial(int vi) const
    {
        return isBitSet(_t[vi], 3);
    }

    INLINE_H_D int getOppTriVi(int vi) const
    {
        if (-1 == _t[vi])
            return -1;

        return getOppValTriVi(_t[vi]);
    }

    INLINE_H_D bool isOppConstraint(int vi) const
    {
        return isOppValConstraint(_t[vi]);
    }

    INLINE_H_D int getOppTri(int vi) const
    {
        return getOppValTri(_t[vi]);
    }

    INLINE_H_D void setOppTri(int vi, int idx)
    {
        return setOppValTri(_t[vi], idx);
    }

    INLINE_H_D int getOppVi(int vi) const
    {
        return getOppValVi(_t[vi]);
    }
};

//////////////////////////////////////////////////////////////////// TriInfo //

// Tri info
// 76543210
//     ^^^^ 0: Dead      1: Alive
//     |||_ 0: Checked   1: Changed
//     ||__ PairType

enum TriCheckState
{
    Checked,
    Changed,
};

enum PairType
{
    PairNone    = 0,
    PairSingle  = 1,
    PairDouble  = 2,
    PairConcave = 3
};

INLINE_H_D bool isTriAlive(char c)
{
    return isBitSet(c, 0);
}

INLINE_H_D void setTriAliveState(char &c, bool b)
{
    setBitState(c, 0, b);
}

INLINE_H_D TriCheckState getTriCheckState(char c)
{
    return isBitSet(c, 1) ? Changed : Checked;
}

INLINE_H_D void setTriCheckState(char &c, TriCheckState s)
{
    if (Checked == s)
        setBitState(c, 1, false);
    else
        setBitState(c, 1, true);
}

INLINE_H_D void setTriPairType(char &c, PairType p)
{
    c = (c & 0xF3) | (p << 2);
}

INLINE_H_D PairType getTriPairType(char c)
{
    return (PairType)((c >> 2) & 3);
}

///////////////////////////////////////////////////////////////////// Constraint //
struct Edge
{
    int _v[2] = {0, 0};

    INLINE_H_D bool operator==(const Edge &edge) const
    {
        return ((_v[0] == edge._v[0]) && (_v[1] == edge._v[1]));
    }

    INLINE_H_D bool operator!=(const Edge &edge) const
    {
        return !(*this == edge);
    }

    INLINE_H_D bool operator<(const Edge &edge) const
    {
        if (_v[0] < edge._v[0])
            return true;
        if (_v[0] > edge._v[0])
            return false;
        if (_v[1] < edge._v[1])
            return true;

        return false;
    }

    INLINE_H_D void sort()
    {
        if (_v[0] > _v[1])
        {
            int tmp = _v[0];
            _v[0]   = _v[1];
            _v[1]   = tmp;
        }
    }
};

///////////////////////////////////////////////////////////////////// Constraint flippint votes //

enum ConsFlipPriority
{
    PriorityCase1 = 0,
    PriorityCase2 = 1,
    PriorityCase3 = 2
};

INLINE_H_D int makeConsFlipVote(int triIdx, ConsFlipPriority priority)
{
    return (priority << 29) | triIdx;
}

INLINE_H_D int getConsFlipVoteIdx(int vote)
{
    return (vote & 0x1FFFFFFF);
}

INLINE_H_D ConsFlipPriority getConsFlipVotePriority(int vote)
{
    return (ConsFlipPriority)(vote >> 29);
}

///////////////////////////////////////////////////////////////////// Flipping //

enum CheckDelaunayMode
{
    CircleFastOrientFast,
    CircleExactOrientSoS
};

enum ActTriMode
{
    ActTriMarkCompact,
    ActTriCollectCompact
};

struct FlipItem
{
    int _v[2];
    int _t[2];
};

struct FlipItemTriIdx
{
    int _t[2];
};

////////////////////////////////////////////////////////// Device containers //

typedef DevVector<bool>     BoolDVec;
typedef DevVector<char>     CharDVec;
typedef DevVector<uchar>    UcharDVec;
using       IntDVec = DevVector<int>;
typedef DevVector<int2>     Int2DVec;
typedef DevVector<float>    FloatDVec;
typedef DevVector<double>   RealDVec;
typedef DevVector<Point>    PointDVec;
typedef DevVector<Edge>     EdgeDVec;
typedef DevVector<Tri>      TriDVec;
typedef DevVector<TriOpp>   TriOppDVec;
typedef DevVector<FlipItem> FlipDVec;

//////////////////////////////////////////////////////////// Host containers //

typedef thrust::host_vector<bool>    BoolHVec;
typedef thrust::host_vector<char>    CharHVec;
typedef thrust::host_vector<int>     IntHVec;
typedef thrust::host_vector<double>  RealHVec;
using Point2DHVec = thrust::host_vector<Point>  ;
using EdgeHVec = thrust::host_vector<Edge> ;

typedef thrust::host_vector<Tri>    TriHVec;
typedef thrust::host_vector<TriOpp> TriOppHVec;

///////////////////////////////////////////////////////////////////// Helper classes //
struct Statistics
{
    double initTime       = .0;
    double splitTime      = .0;
    double flipTime       = .0;
    double relocateTime   = .0;
    double sortTime       = .0;
    double constraintTime = .0;
    double outTime        = .0;
    double totalTime      = .0;

    Statistics() = default;

    void reset()
    {
        initTime       = .0;
        splitTime      = .0;
        flipTime       = .0;
        relocateTime   = .0;
        sortTime       = .0;
        constraintTime = .0;
        outTime        = .0;
        totalTime      = .0;
    }

    void accumulate(const Statistics &s)
    {
        initTime += s.initTime;
        splitTime += s.splitTime;
        flipTime += s.flipTime;
        relocateTime += s.relocateTime;
        sortTime += s.sortTime;
        constraintTime += s.constraintTime;
        outTime += s.outTime;
        totalTime += s.totalTime;
    }

    void average(int div)
    {
        initTime /= div;
        splitTime /= div;
        flipTime /= div;
        relocateTime /= div;
        sortTime /= div;
        constraintTime /= div;
        outTime /= div;
        totalTime /= div;
    }
};

constexpr int TimeLogSize = 6;

class Diagnostic
{
  public:
    int _totFlipNum   = 0;
    int _circleCount  = 0;
    int _rejFlipCount = 0;
    int _flipLoop     = 0;

    double _t[TimeLogSize] = {0, 0, 0, 0, 0, 0};

    void reset()
    {
        _totFlipNum = _flipLoop = _circleCount = _rejFlipCount = 0;

        for (double &i : _t)
            i = 0.0;
    }

    void printCount() const
    {
        std::cout << "  LoopNum: " << _flipLoop << std::endl;
        std::cout << "  FlipNum: " << _totFlipNum << std::endl;
        std::cout << "  Rejected flips: " << _rejFlipCount << std::endl;
        std::cout << "  InCircle check: " << _circleCount << std::endl;

        std::cout << std::endl;
    }

    void printTime()
    {
        std::cout.unsetf(std::ios::floatfield);
        std::cout.setf(std::ios::left, std::ios::adjustfield);

        for (int i = 0; i < TimeLogSize; ++i)
        {
            if (i % 5 == 0)
                std::cout << std::endl << "    ";

            std::cout << "_t" << i << " = ";

            if (_t[i] < 1.0)
                std::cout << "---      ";
            else
                std::cout << std::setw(9) << _t[i];
        }

        std::cout << std::endl;
    }
};

#endif //GDEL2D_COMMONTYPES_H