#ifndef GDEL2D_EDIT_TRIANGULATIONHANDLER_H
#define GDEL2D_EDIT_TRIANGULATIONHANDLER_H

#include "DelaunayChecker.h"
#include "GPU/GpuDelaunay.h"
#include "InputGenerator.h"
#include "PerfTimer.h"
#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <iomanip>

struct line {
    Point2D p1, p2;
};

inline bool onLine(line l1, Point2D p)
{
    // Check whether p is on the line or not
    if (p._p[0] <= std::max(l1.p1._p[0], l1.p2._p[0])
        && p._p[0] <= std::min(l1.p1._p[0], l1.p2._p[0])
        && (p._p[1] <= std::max(l1.p1._p[1], l1.p2._p[1])
            && p._p[1] <= std::min(l1.p1._p[1], l1.p2._p[1])))
        return true;

    return false;
}

inline int direction(Point2D a, Point2D b, Point2D c)
{
    auto val = (b._p[1] - a._p[1]) * (c._p[0] - b._p[0])
               - (b._p[0] - a._p[0]) * (c._p[1] - b._p[1]);

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
    // Four direction for two lines and points of other line
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
    int  runNum   = 1;
    bool doCheck  = false;

    bool        outputResult = false;
    std::string outTriFilename;

    Input        input;
    Output       output;

    Statistics statSum;

    TriangulationHandler() = default;
    void reset();
    void saveResultsToFile();
    bool checkInside(Tri &t, Point2D p) const;

#ifdef WITH_PCL
    static double getupwards(const POINT_TYPE &pt1, const POINT_TYPE &pt2, const POINT_TYPE &pt3);

    static Eigen::Vector3d getTriNormal(const POINT_TYPE &pt1, const POINT_TYPE &pt2, const POINT_TYPE &pt3);

    Eigen::Vector3d getTriNormal(const Tri &t) const;

    static bool hasValidEdge(const POINT_TYPE &pt1, const POINT_TYPE &pt2, const POINT_TYPE &pt3);
#endif
public:
    explicit TriangulationHandler(const char *InputYAMLFile);
    void run();
};

#endif //GDEL2D_EDIT_TRIANGULATIONHANDLER_H