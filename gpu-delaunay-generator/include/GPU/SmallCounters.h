#ifndef GDEL2D_SMALLCOUNTERS_H
#define GDEL2D_SMALLCOUNTERS_H

#include "../CommonTypes.h"
#include "MemoryManager.h"

namespace gdg
{
// Preallocate a collection of small integer counters, initialized to 0.
class SmallCounters
{
  private:
    IntDVec data;
    int     offset = 0;
    int     size   = CounterNum;

  public:
    SmallCounters() = default;
    ~SmallCounters();

    void init();
    void free();
    void renew();
    int *ptr();
    int  operator[](int idx) const;
};
}
#endif //GDEL2D_SMALLCOUNTERS_H