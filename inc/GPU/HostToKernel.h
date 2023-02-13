#ifndef GDEL2D_HOSTTOKERNEL_H
#define GDEL2D_HOSTTOKERNEL_H

#include "../CommonTypes.h"
#include "CudaWrapper.h"

template < typename T >
struct KerArray
{
    T*  _arr;
    int _num;
};

template < typename T >
T* toKernelPtr( DevVector< T >& dVec )
{
    return thrust::raw_pointer_cast( &dVec[0] );
}

template < typename T >
KerArray< T > toKernelArray( DevVector< T >& dVec )
{
    KerArray< T > tArray;
    tArray._arr = toKernelPtr( dVec );
    tArray._num = (int) dVec.size();

    return tArray;
}

typedef KerArray< bool >     KerBoolArray;
typedef KerArray< char >     KerCharArray;
typedef KerArray< uchar >    KerUcharArray;
typedef KerArray< int >      KerIntArray;
typedef KerArray< double > KerRealArray;
typedef KerArray< Point2 >   KerPoint2Array;
typedef KerArray< Tri >      KerTriArray;
typedef KerArray< Segment >  KerSegArray; 
typedef KerArray< TriOpp >   KerOppArray;
typedef KerArray< FlipItem > KerFlipArray; 

#endif //GDEL2D_HOSTTOKERNEL_H