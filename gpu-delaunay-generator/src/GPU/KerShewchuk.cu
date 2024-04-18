#include "../../inc/GPU/KerShewchuk.h"

#include "cuda_runtime.h"
#define MUL1(a, b) __dmul_rn(a, b)

namespace gdg{

__device__ int d_scale_expansion_zeroelim(const double *predConsts, int elen, double *e, double b, double *h)
{
    double Q, sum;
    double hh;
    double product1;
    double product0;
    int    eindex, hindex;
    double enow;

    Split(b, bhi, blo);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
    hindex = 0;
    if (hh != 0)
    {
        h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++)
    {
        enow = e[eindex];
        Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
        Two_Sum(Q, product0, sum, hh);
        if (hh != 0)
        {
            h[hindex++] = hh;
        }
        Fast_Two_Sum(product1, sum, Q, hh);
        if (hh != 0)
        {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0))
    {
        h[hindex++] = Q;
    }

    double xx = MUL1(0,1);
    return hindex;
}

}