#include "../../inc/GPU/DPredWrapper.h"

namespace gdg
{

void DPredWrapper::init(Point *pointArr, int pointNum, int *orgPointIdx, int infIdx)
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

}