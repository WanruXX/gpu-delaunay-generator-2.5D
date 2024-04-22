#ifndef GDEL2D_PCUDAWRAPPER_H
#define GDEL2D_PCUDAWRAPPER_H

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>


#if CUDA_ERROR_CHECK_LEVEL == 0
#define CudaSafeCall(err)
#define CudaCheckError()
#elif CUDA_ERROR_CHECK_LEVEL == 1
#define CudaSafeCall(err) cuda_safe_call(err, __FILE__, __LINE__)
#define CudaCheckError() cuda_check_error_loose(__FILE__, __LINE__)
#else
#define CudaSafeCall(err) cuda_safe_call(err, __FILE__, __LINE__)
#define CudaCheckError cuda_check_error(__FILE__, __LINE__)
#endif

#if CUDA_ERROR_CHECK_LEVEL > 0

inline void cuda_safe_call(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err)
    {
        std::cerr << "cudaSafeCall() failed at " << file << ": " << line << ": " << cudaGetErrorString(err)
                  << std::endl;
        exit(-1);
    }
#endif
}

inline void cuda_check_error(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%d_i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }

    // More careful checking. However, this will affect performance.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%d_i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

inline void cuda_check_error_loose(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%d_i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

#if __CUDA_ARCH__ >= 200 && defined(CUDA_ERROR_CHECK)
#define CudaAssert(X)                                                                                                  \
    if (!(X))                                                                                                          \
    {                                                                                                                  \
        printf("!!!Thread %d:%d failed assert at %s:%d!!!\n", blockIdx.x, threadIdx.x, __FILE__, __LINE__);            \
    }
#else
#define CudaAssert(X)
#endif
#endif

namespace gdg
{
template <typename T>
T *cuNew(int num)
{
    T           *loc   = NULL;
    const size_t space = num * sizeof(T);
    CudaSafeCall(cudaMalloc(&loc, space));

    return loc;
}

template <typename T>
void cuDelete(T **loc)
{
    CudaSafeCall(cudaFree(*loc));
    *loc = NULL;
}

template <typename T>
__forceinline__ __device__ void cuSwap(T &v0, T &v1)
{
    const T tmp = v0;
    v0          = v1;
    v1          = tmp;
}

inline void cuPrintMemory(const char *inStr)
{
    const int MegaByte = (1 << 20);

    size_t free;
    size_t total;
    CudaSafeCall(cudaMemGetInfo(&free, &total));
    std::cout << inStr << " Memory used: " << (total - free) / MegaByte << " MB" << std::endl;
}

// Obtained from: C:\ProgramData\NVIDIA Corporation\GPU SDK\C\common\inc\cutil_inline_runtime.h
// This function returns the best GPU (with maximum GFLOPS)
inline int cutGetMaxGflopsDeviceId()
{
    int            current_device = 0;
    int            sm_per_multiproc;
    int            max_compute_perf = 0, max_perf_device = 0;
    int            device_count = 0, best_SM_arch = 0;
    int            arch_cores_sm[3] = {1, 8, 32};
    cudaDeviceProp deviceProp{};

    cudaGetDeviceCount(&device_count);
    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);
        if (deviceProp.major > 0 && deviceProp.major < 9999)
        {
            if (deviceProp.major > best_SM_arch)
                best_SM_arch = deviceProp.major;
        }
        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;
    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        {
            sm_per_multiproc = 1;
        }
        else if (deviceProp.major <= 2)
        {
            sm_per_multiproc = arch_cores_sm[deviceProp.major];
        }
        else
        {
            sm_per_multiproc = arch_cores_sm[2];
        }

        int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        if (compute_perf > max_compute_perf)
        {
            // If we find GPU with SM major > 2, search only these
            if (best_SM_arch > 2)
            {
                // If our device==dest_SM_arch, choose this, or else pass
                if (deviceProp.major == best_SM_arch)
                {
                    max_compute_perf = compute_perf;
                    max_perf_device  = current_device;
                }
            }
            else
            {
                max_compute_perf = compute_perf;
                max_perf_device  = current_device;
            }
        }
        ++current_device;
    }
    return max_perf_device;
}
}

#endif //GDEL2D_PCUDAWRAPPER_H