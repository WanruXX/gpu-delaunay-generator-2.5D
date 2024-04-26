#include "inc/TriangulationHandler.h"
#include <unistd.h>

int main(int argc, char *argv[])
{
    int deviceCount;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
    {
        std::cerr << "Cuda: cannot get device count!" << std::endl;
        return -1;
    }
    cudaDeviceProp properties{};
    cudaResultCode = cudaGetDeviceProperties(&properties, 0);
    if (cudaResultCode != cudaSuccess || properties.major == 9999)
    {
        std::cerr << "Cuda: cannot find main device (device0) or the main device is emulation only!" << std::endl;
        return -1;
    }
    printf("multiProcessorCount %d\n",properties.multiProcessorCount);
    printf("maxThreadsPerMultiProcessor %d\n",properties.maxThreadsPerMultiProcessor);

    if (argc != 2)
    {
        std::cerr << "Usage: ./delaunay_generator config.yaml" << std::endl;
        return -1;
    }

    const char *YAMLFile(argv[1]);
    if (access(YAMLFile, F_OK) == -1)
    {
        std::cerr << YAMLFile << " is not a valid path!" << std::endl;
        return -1;
    }

    TriangulationHandler app(YAMLFile);
    app.run();
}
