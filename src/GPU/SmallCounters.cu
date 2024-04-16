#include "../../inc/GPU/HostToKernel.h"
#include "../../inc/GPU/SmallCounters.h"

namespace
{
constexpr std::size_t CounterCapacity = 1024 * 8;
}

void SmallCounters::init()
{
    data.assign(CounterCapacity, 0);
}

SmallCounters::~SmallCounters()
{
    free();
}

void SmallCounters::free()
{
    data.free();
}

void SmallCounters::renew()
{
    if (data.size() == 0)
    {
        printf("Flag not initialized!\n");
        exit(-1);
    }

    offset += size;

    if (offset + size > data.capacity())
    {
        offset = 0;
        data.fill(0);
    }
}

int *SmallCounters::ptr()
{
    return toKernelPtr(data) + offset;
}

int SmallCounters::operator[](int idx) const
{
    assert(idx < size);

    return data[offset + idx];
}
