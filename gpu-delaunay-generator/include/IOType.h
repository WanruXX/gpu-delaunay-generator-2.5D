#ifndef DELAUNAY_GENERATOR_IOTYPE_H
#define DELAUNAY_GENERATOR_IOTYPE_H

#include "CommonTypes.h"

namespace gdg
{

struct Input
{
    Point2DHVec pointVec;
    EdgeHVec    constraintVec;

    bool      insAll    = false; // Insert all before flipping
    bool      noSort    = false; // Sort input points (unused)
    bool      noReorder = false; // Reorder the triangle before flipping

    Input() = default;

    void removeDuplicates();
};

struct Output
{
    TriHVec        triVec;
    TriOppHVec     triOppVec;
    std::set<Edge> edgeSet;
    Point          infPt;

    void reset();

    void getEdgesFromTriVec();
};
}
#endif //DELAUNAY_GENERATOR_IOTYPE_H