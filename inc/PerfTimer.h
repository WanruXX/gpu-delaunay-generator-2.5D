/*
Author: Cao Thanh Tung, Ashwin Nanjappa
Date:   05-Aug-2014

===============================================================================

Copyright (c) 2011, School of Computing, National University of Singapore. 
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/gdel3d.html

If you use gDel3D and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the National University of University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef GDEL2D_PERFTIMER_H
#define GDEL2D_PERFTIMER_H

#include "GPU/CudaWrapper.h"
#include <sys/time.h>

const long long NANO_PER_SEC = 1000000000LL;
const long long MICRO_TO_NANO = 1000LL;

struct PerfTimer
{
    long long _startTime = 0;
    long long _stopTime = 0;
    long long _leftover = 0;
    long long _value = 0;

    PerfTimer() = default;

    static long long _getTime()
    {
        struct timeval tv{};
        long long ntime;

        if (0 == gettimeofday(&tv, nullptr))
        {
            ntime  = NANO_PER_SEC;
            ntime *= tv.tv_sec;
            ntime += tv.tv_usec * MICRO_TO_NANO;
        }

        return ntime;
    }

    virtual void start()
    {
        _startTime = _getTime();
    }

    virtual void stop()
    {
        _stopTime = _getTime();
        _value      = _leftover + _stopTime - _startTime; 
        _leftover   = 0; 
    }

    virtual void pause()
    {
        _stopTime   = _getTime();
        _leftover   += _stopTime - _startTime;         
    }

    double value() const
    {
        return ((double) _value) / NANO_PER_SEC * 1000;
    }
};

class CudaTimer : public PerfTimer
{
public:
    void start() override
    {
        CudaSafeCall(cudaDeviceSynchronize());
        PerfTimer::start();
    }

    void stop() override
    {
        CudaSafeCall(cudaDeviceSynchronize());
        PerfTimer::stop();
    }

    void pause() override
    {
        CudaSafeCall(cudaDeviceSynchronize());
        PerfTimer::pause();
    }

    double value()
    {
        return PerfTimer::value();
    }
};

#endif //GDEL2D_PERFTIMER_H
