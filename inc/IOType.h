#ifndef DELAUNAY_GENERATOR_IOTYPE_H
#define DELAUNAY_GENERATOR_IOTYPE_H

#include "CommonTypes.h"

// Different level of code profiling
enum ProfLevel
{
    ProfNone,
    ProfDefault,
    ProfDetail,
    ProfDiag,
    ProfDebug,
    ProfLevelCount
};

struct Input
{
    Point2DHVec  pointVec;
    EdgeHVec     constraintVec;

    bool      insAll    = false; // Insert all before flipping
    bool      noSort    = false; // Sort input points (unused)
    bool      noReorder = false; // Reorder the triangle before flipping
    ProfLevel profLevel = ProfDefault;

    bool isProfiling(ProfLevel level) const
    {
        return (profLevel >= level);
    }

    Input() = default;

    void removeDuplicates();
};

struct Output
{
    TriHVec           triVec;
    TriOppHVec        triOppVec;
    std::set<Edge> segVec;
    Point             ptInf;
    Statistics        stats;
};

#endif //DELAUNAY_GENERATOR_IOTYPE_H
