#include "inc/TriangulationHandler.h"
#include <unistd.h>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: ./gdel2d_edit config.yaml" << std::endl;
        return -1;
    }
    const char *YAMLFile(argv[1]);

    if (access(YAMLFile, F_OK) == -1)
    {
        std::cerr << YAMLFile << " is not a valid path!" << std::endl;
    }
    else
    {
        TriangulationHandler app(YAMLFile);
        app.run();
    }
}
