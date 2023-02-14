#include "DPredWrapper.h"
#include "KerShewchuk.h"

void DPredWrapper::init(Point2 *pointArr, int pointNum, int *orgPointIdx, int infIdx)
{
    _pointArr    = pointArr;
    _pointNum    = pointNum;
    _orgPointIdx = orgPointIdx;
    _infIdx      = infIdx;

    _predConsts = cuNew<double>(DPredicateBoundNum);

    kerInitPredicate<<<1, 1>>>(_predConsts);
    CudaCheckError();
}

void DPredWrapper::cleanup()
{
    cuDelete(&_predConsts);
}

__forceinline__ __device__ const Point2 &DPredWrapper::getPoint(int idx) const
{
    return _pointArr[idx];
}

__forceinline__ __device__ int DPredWrapper::getPointIdx(int idx) const
{
    return (_orgPointIdx == nullptr) ? idx : _orgPointIdx[idx];
}

__forceinline__ __device__ __host__ int DPredWrapper::pointNum() const
{
    return _pointNum;
}

__forceinline__ __device__ Orient DPredWrapper::doOrient2DFast(int v0, int v1, int v2) const
{
    const double *pt[] = {getPoint(v0)._p, getPoint(v1)._p, getPoint(v2)._p};

    double det = orient2dFast(_predConsts, pt[0], pt[1], pt[2]);

    //CudaAssert( v2 != _infIdx );

    if (v0 == _infIdx | v1 == _infIdx | v2 == _infIdx)
        det = -det;

    return ortToOrient(det);
}

__forceinline__ __device__ Orient DPredWrapper::doOrient2DFastExact(const double *p0,
                                                                    const double *p1,
                                                                    const double *p2) const
{
    const double det = orient2dFastExact(_predConsts, p0, p1, p2);
    return ortToOrient(det);
}

__forceinline__ __device__ Orient
DPredWrapper::doOrient2DSoSOnly(const double *p0, const double *p1, const double *p2, int v0, int v1, int v2)
{
    ////
    // Sort points using vertex as key, also note their sorted order
    ////
    const double *p[DEG] = {p0, p1, p2};
    int           pn     = 1;

    if (v0 > v1)
    {
        cuSwap(v0, v1);
        cuSwap(p[0], p[1]);
        pn = -pn;
    }
    if (v0 > v2)
    {
        cuSwap(v0, v2);
        cuSwap(p[0], p[2]);
        pn = -pn;
    }
    if (v1 > v2)
    {
        cuSwap(v1, v2);
        cuSwap(p[1], p[2]);
        pn = -pn;
    }

    double result = 0;
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

__forceinline__ __device__ Orient DPredWrapper::doOrient2DFastExactSoS(int v0, int v1, int v2) const
{
    const double *pt[] = {getPoint(v0)._p, getPoint(v1)._p, getPoint(v2)._p};

    // Fast-Exact
    Orient ord = doOrient2DFastExact(pt[0], pt[1], pt[2]);

    if (OrientZero == ord)
    {
        // SoS
        if (_orgPointIdx != nullptr)
        {
            v0 = _orgPointIdx[v0];
            v1 = _orgPointIdx[v1];
            v2 = _orgPointIdx[v2];
        }

        ord = doOrient2DSoSOnly(pt[0], pt[1], pt[2], v0, v1, v2);
    }

    //CudaAssert( v2 != _infIdx );

    if ((v0 == _infIdx) | (v1 == _infIdx) | (v2 == _infIdx))
        ord = flipOrient(ord);

    return ord;
}

/////////////////////////////////////////////////////////////////// InCircle //

__forceinline__ __device__ Side DPredWrapper::doInCircleFast(Tri tri, int vert) const
{
    const double *pt[] = {getPoint(tri._v[0])._p, getPoint(tri._v[1])._p, getPoint(tri._v[2])._p, getPoint(vert)._p};

    if (vert == _infIdx)
        return SideOut;

    double det;

    if (tri.has(_infIdx))
    {
        const int infVi = tri.getIndexOf(_infIdx);

        det = orient2dFast(_predConsts, pt[(infVi + 1) % 3], pt[(infVi + 2) % 3], pt[3]);
    }
    else
        det = incircleFast(_predConsts, pt[0], pt[1], pt[2], pt[3]);

    return cicToSide(det);
}

__forceinline__ __device__ Side DPredWrapper::doInCircleFastExact(const double *p0,
                                                                  const double *p1,
                                                                  const double *p2,
                                                                  const double *p3) const
{
    double det = incircleFastAdaptExact(_predConsts, p0, p1, p2, p3);

    return cicToSide(det);
}

__forceinline__ __device__ double DPredWrapper::doOrient1DExact_Lifted(const double *p0, const double *p1) const
{
    const double det = orient1dExact_Lifted(_predConsts, p0, p1);
    return det;
}

__forceinline__ __device__ double
DPredWrapper::doOrient2DExact_Lifted(const double *p0, const double *p1, const double *p2, bool lifted) const
{
    const double det = orient2dExact_Lifted(_predConsts, p0, p1, p2, lifted);
    return det;
}

// Exact Incircle check must have failed (d_i.e. returned 0)
// No Infinity point here!!!
__forceinline__ __device__ Side DPredWrapper::doInCircleSoSOnly(const double *p0,
                                                                const double *p1,
                                                                const double *p2,
                                                                const double *p3,
                                                                int           v0,
                                                                int           v1,
                                                                int           v2,
                                                                int           v3) const
{
    ////
    // Sort points using vertex as key, also note their sorted order
    ////

    static int    call_num = 0;
    const int     NUM      = DEG + 1;
    const double *p[NUM]   = {p0, p1, p2, p3};
    int           pn       = 1;

    if (v0 > v2)
    {
        cuSwap(v0, v2);
        cuSwap(p[0], p[2]);
        pn = -pn;
    }
    if (v1 > v3)
    {
        cuSwap(v1, v3);
        cuSwap(p[1], p[3]);
        pn = -pn;
    }
    if (v0 > v1)
    {
        cuSwap(v0, v1);
        cuSwap(p[0], p[1]);
        pn = -pn;
    }
    if (v2 > v3)
    {
        cuSwap(v2, v3);
        cuSwap(p[2], p[3]);
        pn = -pn;
    }
    if (v1 > v2)
    {
        cuSwap(v1, v2);
        cuSwap(p[1], p[2]);
        pn = -pn;
    }

    double result = 0;
    double pa2[2], pb2[2], pc2[2];
    int    depth;

    for (depth = 0; depth < 14; ++depth)
    {
        bool lifted = false;
        //        if(depth > 4){
        //            printf("call %d, depth: %d\n", call_num,  depth);
        //        }

        switch (depth)
        {
        case 0:
            //printf("Here %d_i", depth);
            pa2[0] = p[1][0];
            pa2[1] = p[1][1];
            pb2[0] = p[2][0];
            pb2[1] = p[2][1];
            pc2[0] = p[3][0];
            pc2[1] = p[3][1];
            break;
        case 1:
            lifted = true;
            //printf("Here %d_i", depth);
            //pa2[0] = p[1][0];   pa2[1] = p[1][1];
            //pb2[0] = p[2][0];   pb2[1] = p[2][1];
            //pc2[0] = p[3][0];   pc2[1] = p[3][1];
            break;
        case 2:
            lifted = true;
            //printf("Here %d_i", depth);
            pa2[0] = p[1][1];
            pa2[1] = p[1][0];
            pb2[0] = p[2][1];
            pb2[1] = p[2][0];
            pc2[0] = p[3][1];
            pc2[1] = p[3][0];
            break;
        case 3:
            //printf("Here %d_i", depth);
            pa2[0] = p[0][0];
            pa2[1] = p[0][1];
            pb2[0] = p[2][0];
            pb2[1] = p[2][1];
            pc2[0] = p[3][0];
            pc2[1] = p[3][1];
            break;
        case 4:
            //printf("Here %d_i", depth);
            result = p[2][0] - p[3][0];
            break;
        case 5:
            //printf("Here %d_i", depth);
            result = p[2][1] - p[3][1];
            break;
        case 6:
            lifted = true;
            // printf("Here %d_i\n", depth);
            //pa2[0] = p[0][0];   pa2[1] = p[0][1];
            //pb2[0] = p[2][0];   pb2[1] = p[2][1];
            //pc2[0] = p[3][0];   pc2[1] = p[3][1];
            break;
        case 7:
            lifted = true;
            //printf("Here %d_i\n", depth);
            pa2[0] = p[2][0];
            pa2[1] = p[2][1];
            pb2[0] = p[3][0];
            pb2[1] = p[3][1];
            break;
        case 8:
            lifted = true;
            // printf("Here %d_i\n", depth);
            pa2[0] = p[0][1];
            pa2[1] = p[0][0];
            pb2[0] = p[2][1];
            pb2[1] = p[2][0];
            pc2[0] = p[3][1];
            pc2[1] = p[3][0];
            break;
        case 9:
            //printf("Here %d_i\n", depth);
            pa2[0] = p[0][0];
            pa2[1] = p[0][1];
            pb2[0] = p[1][0];
            pb2[1] = p[1][1];
            pc2[0] = p[3][0];
            pc2[1] = p[3][1];
            break;
        case 10:
            //printf("Here %d_i\n", depth);
            result = p[1][0] - p[3][0];
            break;
        case 11:
            // printf("Here %d_i\n", depth);
            result = p[1][1] - p[3][1];
            break;
        case 12:
            //printf("Here %d_i\n", depth);
            result = p[0][0] - p[3][0];
            break;
        default:
            // printf("Here %d_i\n", depth);
            result = 1.0;
            break;
        }

        switch (depth)
        {
        // 2D orientation determinant
        case 0:
        case 3:
        case 9:
            // 2D orientation involving the lifted coordinate
        case 1:
        case 2:
        case 6:
        case 8:
            result = doOrient2DExact_Lifted(pa2, pb2, pc2, lifted);
            break;
            // 1D orientation involving the lifted coordinate
        case 7:
            result = doOrient1DExact_Lifted(pa2, pb2);
            break;
        default:
            break;
        }

        if (result != 0)
            break;
    }

    switch (depth)
    {
    case 1:
    case 3:
    case 5:
    case 8:
    case 10:
        result = -result;
        break;
    default:
        break;
    }

    const double det = result * pn;

    ++call_num;

    return cicToSide(det);
}

__forceinline__ __device__ Side DPredWrapper::doInCircleFastExactSoS(Tri tri, int vert) const
{
    if (vert == _infIdx)
        return SideOut;

    const double *pt[] = {getPoint(tri._v[0])._p, getPoint(tri._v[1])._p, getPoint(tri._v[2])._p, getPoint(vert)._p};

    if (tri.has(_infIdx))
    {
        const int infVi = tri.getIndexOf(_infIdx);

        const Orient ort = doOrient2DFastExactSoS(tri._v[(infVi + 1) % 3], tri._v[(infVi + 2) % 3], vert);

        return cicToSide(ort);
    }

    const Side s0 = doInCircleFastExact(pt[0], pt[1], pt[2], pt[3]);

    if (SideZero != s0)
        return s0;

    // SoS
    if (_orgPointIdx != nullptr)
    {
        tri._v[0] = _orgPointIdx[tri._v[0]];
        tri._v[1] = _orgPointIdx[tri._v[1]];
        tri._v[2] = _orgPointIdx[tri._v[2]];
        vert      = _orgPointIdx[vert];
    }

    const Side s1 = doInCircleSoSOnly(pt[0], pt[1], pt[2], pt[3], tri._v[0], tri._v[1], tri._v[2], vert);

    return s1;
}

__forceinline__ __device__ double DPredWrapper::inCircleDet(Tri tri, int vert) const
{
    const double *pt[] = {getPoint(tri._v[0])._p, getPoint(tri._v[1])._p, getPoint(tri._v[2])._p, getPoint(vert)._p};

    double det;

    if (tri.has(_infIdx))
    {
        const int infVi = tri.getIndexOf(_infIdx);

        det = orient2dDet(pt[(infVi + 1) % 3], pt[(infVi + 2) % 3], pt[3]);
    }
    else
        det = incircleDet(pt[0], pt[1], pt[2], pt[3]);

    return det;
}