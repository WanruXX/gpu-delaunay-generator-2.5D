#ifndef DELAUNAY_GENERATOR_TRIANGULATIONHANDLER_H
#define DELAUNAY_GENERATOR_TRIANGULATIONHANDLER_H

#include "gpu-delaunay-generator.h"
#include "InputGenerator.h"
#include <bits/stdc++.h>
#include <iomanip>

struct line
{
    gdg::Point p1, p2;
};

inline bool onLine(line l1, gdg::Point p)
{
    // Check whether p is on the line or not
    if (p._p[0] <= std::max(l1.p1._p[0], l1.p2._p[0]) && p._p[0] <= std::min(l1.p1._p[0], l1.p2._p[0]) &&
        (p._p[1] <= std::max(l1.p1._p[1], l1.p2._p[1]) && p._p[1] <= std::min(l1.p1._p[1], l1.p2._p[1])))
        return true;

    return false;
}

inline int direction(gdg::Point a, gdg::Point b, gdg::Point c)
{
    auto val = (b._p[1] - a._p[1]) * (c._p[0] - b._p[0]) - (b._p[0] - a._p[0]) * (c._p[1] - b._p[1]);

    if (val == 0)

        // Colinear
        return 0;

    else if (val < 0)

        // Anti-clockwise direction
        return 2;

    // Clockwise direction
    return 1;
}

inline bool isIntersect(line l1, line l2)
{
    // Four direction for two lines and gdg::Points of other line
    int dir1 = direction(l1.p1, l1.p2, l2.p1);
    int dir2 = direction(l1.p1, l1.p2, l2.p2);
    int dir3 = direction(l2.p1, l2.p2, l1.p1);
    int dir4 = direction(l2.p1, l2.p2, l1.p2);

    // When intersecting
    if (dir1 != dir2 && dir3 != dir4)
        return true;

    // When p2 of line2 are on the line1
    if (dir1 == 0 && onLine(l1, l2.p1))
        return true;

    // When p1 of line2 are on the line1
    if (dir2 == 0 && onLine(l1, l2.p2))
        return true;

    // When p2 of line1 are on the line2
    if (dir3 == 0 && onLine(l2, l1.p1))
        return true;

    // When p1 of line1 are on the line2
    if (dir4 == 0 && onLine(l2, l1.p2))
        return true;

    return false;
}

class TriangulationHandler
{
  private:
    TriangulationHandler() = default;

    void reset();
    void saveResultsToFile();
    void saveToGeojson(std::ofstream &outputTri) const;
    void saveToObj(std::ofstream &outputTri) const;
    bool checkInside(gdg::Tri &t, gdg::Point p) const;

    int         runNum       = 1;
    bool        doCheck      = false;
    bool        outputResult = false;
    std::string OutputFilename;

    gdg::Input       input;
    gdg::Output      output;
    gdg::Statistics  statSum;

  public:
    explicit TriangulationHandler(const char *InputYAMLFile);
    void run();
};

#endif //DELAUNAY_GENERATOR_TRIANGULATIONHANDLER_H