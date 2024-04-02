#ifndef GDEL2D_HASHTABLE_H
#define GDEL2D_HASHTABLE_H

#include "CommonTypes.h"

struct Point2DHash
{
    std::size_t operator()(const Point2D &p) const
    {
        auto h1 = std::hash<double>{}(p._p[0]);
        auto h2 = std::hash<double>{}(p._p[1]);
        return h1 ^ h2;
    }
};

struct EdgeHash
{
    std::size_t operator()(const Edge &edge) const
    {
        auto h1 = std::hash<int>{}(edge._v[0]);
        auto h2 = std::hash<int>{}(edge._v[1]);
        return h1 ^ h2;
    }
};

struct EdgeEqual
{
    bool operator()(const Edge &edge1, const Edge &edge2) const
    {
        return (edge1._v[0] == edge2._v[0] && edge1._v[1] == edge2._v[1]) || (edge1._v[0] == edge2._v[1] && edge1._v[1] == edge2._v[0]);
    }
};


#endif //GDEL2D_HASHTABLE_H
