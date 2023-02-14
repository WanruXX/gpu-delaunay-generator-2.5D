#include "../inc/HashTable.h"

int HashUInt::operator()(unsigned int x) const
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return static_cast<int>(x);
}

int HashPoint2::operator()(Point2 p) const
{
    auto x = static_cast<unsigned int>(p._p[0]);
    auto y = static_cast<unsigned int>(p._p[1]);

    return hashUInt(x) + hashUInt(y);
}