#include "../../inc/CPU/PredWrapper.h"

PredWrapper::PredWrapper() : _pointArr(nullptr), _pointNum(0), _infIdx(0)
{
}

PredWrapper::PredWrapper(const Point2DHVec &pointVec, Point ptInfty)
{
    _pointArr = &pointVec[0];
    _pointNum = pointVec.size();
    _infIdx   = _pointNum;
    _ptInfty  = ptInfty;
    exactinit();
}

const Point &PredWrapper::getPoint(int idx) const
{
    return (idx == _infIdx) ? _ptInfty : _pointArr[idx];
}

size_t PredWrapper::pointNum() const
{
    return _pointNum + 1;
}

Orient PredWrapper::doOrient2D(int v0, int v1, int v2) const
{
    assert((v0 != v1) && (v0 != v2) && (v1 != v2) && "Duplicate indices in orientation!");

    const Point p[] = {getPoint(v0), getPoint(v1), getPoint(v2)};

    double det = orient2d(p[0]._p, p[1]._p, p[2]._p);

    if ((v0 == _infIdx) || (v1 == _infIdx) || (v2 == _infIdx))
        det = -det;

    return ortToOrient(det);
}

Orient PredWrapper::doOrient2DSoSOnly(const double *p0, const double *p1, const double *p2, int v0, int v1, int v2)
{
    ////
    // Sort points using vertex as key, also note their sorted order
    ////
    const double *p[DEG] = {p0, p1, p2};
    int           pn     = 1;

    if (v0 > v1)
    {
        std::swap(v0, v1);
        std::swap(p[0], p[1]);
        pn = -pn;
    }
    if (v0 > v2)
    {
        std::swap(v0, v2);
        std::swap(p[0], p[2]);
        pn = -pn;
    }
    if (v1 > v2)
    {
        std::swap(v1, v2);
        std::swap(p[1], p[2]);
        pn = -pn;
    }

    double result;
    int    depth;

    for (depth = 1; depth <= 4; ++depth)
    {
        switch (depth)
        {
        case 1:
            result = p[2][0] - p[1][0];
            break;
        case 2:
            result = p[1][1] - p[2][1];
            break;
        case 3:
            result = p[0][0] - p[2][0];
            break;
        default:
            result = 1.0;
            break;
        }

        if (result != 0)
            break;
    }

    const double det = result * pn;

    return ortToOrient(det);
}

Orient PredWrapper::doOrient2DFastExactSoS(int v0, int v1, int v2) const
{
    const double *pt[] = {getPoint(v0)._p, getPoint(v1)._p, getPoint(v2)._p};

    // Fast-Exact
    Orient ord = doOrient2D(v0, v1, v2);

    if (OrientZero == ord)
        ord = doOrient2DSoSOnly(pt[0], pt[1], pt[2], v0, v1, v2);

    if (v0 == _infIdx | v1 == _infIdx)
        ord = flipOrient(ord);

    return ord;
}

///////////////////////////////////////////////////////////////////// Circle //

Side PredWrapper::doIncircle(Tri tri, int vert) const
{
    if (vert == _infIdx)
        return SideOut;

    const Point pt[] = {getPoint(tri._v[0]), getPoint(tri._v[1]), getPoint(tri._v[2]), getPoint(vert)};

    double det;

    const auto int_infIdx = static_cast<int>(_infIdx);
    if (tri.has(int_infIdx))
    {
        const int infVi = tri.getIndexOf(int_infIdx);

        det = orient2d(pt[(infVi + 1) % 3]._p, pt[(infVi + 2) % 3]._p, pt[3]._p);
    }
    else
        det = incircle(pt[0]._p, pt[1]._p, pt[2]._p, pt[3]._p);

    return cicToSide(det);
}
