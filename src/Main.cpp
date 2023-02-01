#define WINDOWS_VISUALIZATION

#include <iomanip>

#include "gDel2D/GpuDelaunay.h"
#include "gDel2D/PerfTimer.h"

#include "gDel2D/CPU/PredWrapper.h"

#include "InputCreator.h"
#include "DelaunayChecker.h"

#if defined(_WIN32)
#include <Windows.h>
#endif

#ifdef WINDOWS_VISUALIZATION

#include "Visualizer.h"

#endif

///////////////////////////////////////////////////////////////////////////////

Point2HVec PointVecCopy;

class App {
public:
    // Parameters
    Distribution _dist;
    int _seed;
    bool _doCheck;      // Check Euler, orientation, etc.
    bool _inFile;       // Input from file
    std::string _inFilename;

    // In-Out Data
    GDel2DInput _input;
    GDel2DOutput _output;

    // Main
    int _runNum;
    int _pointNum;
    int _constraintNum;

    // Statistics
    Statistics statSum;

public:

    App() {
        _pointNum = 1000;
        _constraintNum = -1;
        _dist = UniformDistribution;
        _seed = 76213898;
        _inFile = false;
        _doCheck = false;

        _runNum = 1;

        return;
    }

    void reset() {
        Point2HVec().swap(_input.pointVec);
        SegmentHVec().swap(_input.constraintVec);
        TriHVec().swap(_output.triVec);
        TriOppHVec().swap(_output.triOppVec);

        cudaDeviceReset();

        return;
    }

    void run() {
        // Pick the best CUDA device
        const int deviceIdx = cutGetMaxGflopsDeviceId();
        CudaSafeCall(cudaSetDevice(deviceIdx));
        CudaSafeCall(cudaDeviceReset());

        GpuDel gpuDel;

        for (int i = 0; i < _runNum; ++i) {
            reset();

            std::cout << "Point set: " << _seed << std::endl;

            // 1. Create points
            InputCreator creator;

            if (_inFile) {
                creator.readPoints(_inFilename, _input.pointVec, _input.constraintVec, _constraintNum);
                PointVecCopy = _input.pointVec;

                _pointNum = _input.pointVec.size();
            } else
                creator.makePoints(_pointNum, _dist, _input.pointVec, _seed);

            // 2. Compute Delaunay triangulation
            gpuDel.compute(_input, &_output);

            const Statistics &stats = _output.stats;

            statSum.accumulate(stats);

            std::cout << "TIME: " << stats.totalTime << "("
                      << stats.initTime << ", "
                      << stats.splitTime << ", "
                      << stats.flipTime << ", "
                      << stats.relocateTime << ", "
                      << stats.sortTime << ", "
                      << stats.constraintTime << ", "
                      << stats.outTime << ")"
                      << std::endl;

            //////////////

            if (_doCheck) {
                DelaunayChecker checker(_input, _output);

                std::cout << "\n*** Check ***\n";

                checker.checkEuler();
                checker.checkOrientation();
                checker.checkAdjacency();
                checker.checkConstraints();
                checker.checkDelaunay();
//                checker.outputSegments("/home/wxxs6p/Downloads/gDel2D-Oct2015/Fede_left_4_segm.geojson", _input.pointVec);

                std::ifstream InputPCStream("/home/wxxs6p/Downloads/gDel2D-Oct2015/Fede_left_4.txt");
                std::string LineData;
                double data;
                std::getline(InputPCStream, LineData);
                std::vector<std::pair<double, double>> InputPC;
                while (std::getline(InputPCStream, LineData)) {
                    InputPC.emplace_back();
                    std::stringstream ss(LineData);
                    ss >> data;
                    ss >> data;
                    InputPC.back().first = data;
                    ss >> data;
                    InputPC.back().second = data;
                }

                std::cout << InputPC.size() << " " << _input.pointVec.size() << std::endl;
                int NumChaned = 0;
                for (std::size_t ii = 0; ii < PointVecCopy.size(); ++ii) {
//                    if(PointVecCopy[ii]._p[0] != _input.pointVec[ii]._p[0] || PointVecCopy[ii]._p[1] != _input.pointVec[ii]._p[1]){
                    if (InputPC[ii].first != _input.pointVec[ii]._p[0] ||
                        InputPC[ii].second != _input.pointVec[ii]._p[1]) {
//                        std::cout << "point changes at " << ii << std::endl;
                        ++NumChaned;
                    }
                }
                std::cout << "changed pt num " << NumChaned << std::endl;
            }

            ++_seed;
        }

    }

};

void summarize(const App &app) {
    ////
    // Summarize on screen
    ////

    Statistics stats = app.statSum;

    stats.average(app._runNum);

    std::cout << std::endl;
    std::cout << "---- SUMMARY ----" << std::endl;
    std::cout << std::endl;

    std::cout << "PointNum       " << app._pointNum << std::endl;
    std::cout << "Input          " << DistStr[app._dist] << std::endl;
    std::cout << "FP Mode        " << ((sizeof(RealType) == 8) ? "Double" : "Single") << std::endl;
    std::cout << "Sort           " << (app._input.noSort ? "no" : "yes") << std::endl;
    std::cout << "Reorder        " << (app._input.noReorder ? "no" : "yes") << std::endl;
    std::cout << "Insert mode    " << (app._input.insAll ? "InsAll" : "InsFlip") << std::endl;
    std::cout << std::endl;
    std::cout << std::fixed << std::right << std::setprecision(2);
    std::cout << "TotalTime (ms) " << std::setw(10) << stats.totalTime << std::endl;
    std::cout << "InitTime       " << std::setw(10) << stats.initTime << std::endl;
    std::cout << "SplitTime      " << std::setw(10) << stats.splitTime << std::endl;
    std::cout << "FlipTime       " << std::setw(10) << stats.flipTime << std::endl;
    std::cout << "RelocateTime   " << std::setw(10) << stats.relocateTime << std::endl;
    std::cout << "SortTime       " << std::setw(10) << stats.sortTime << std::endl;
    std::cout << "ConstraintTime " << std::setw(10) << stats.constraintTime << std::endl;
    std::cout << "OutTime        " << std::setw(10) << stats.outTime << std::endl;
    std::cout << std::endl;

    ////
    // Write to file
    ////
    const std::string fname = "/home/wxxs6p/Downloads/gDel2D-Oct2015/statistics.csv";

    std::ofstream outFile(fname.c_str(), std::ofstream::app);

    if (!outFile) {
        std::cerr << "Error opening output file!\n";
        exit(1);
    }

    outFile << "GridWidth," << GridSize << ",";
    outFile << "PointNum," << app._pointNum << ",";
    outFile << "Runs," << app._runNum << ",";
    outFile << "Input," << (app._inFile ? app._inFilename : DistStr[app._dist]) << ",";
    outFile << (app._input.noSort ? "--" : "sort") << ",";
    outFile << (app._input.noReorder ? "--" : "reorder") << ",";
    outFile << (app._input.insAll ? "InsAll" : "--") << ",";

    outFile << "TotalTime," << stats.totalTime / 1000.0 << ",";
    outFile << "InitTime," << stats.initTime / 1000.0 << ",";
    outFile << "SplitTime," << stats.splitTime / 1000.0 << ",";
    outFile << "FlipTime," << stats.flipTime / 1000.0 << ",";
    outFile << "RelocateTime," << stats.relocateTime / 1000.0 << ",";
    outFile << "SortTime," << stats.sortTime / 1000.0 << ",";
    outFile << "ConstraintTime," << stats.constraintTime / 1000.0 << ",";
    outFile << "OutTime," << stats.outTime / 1000.0;
    outFile << std::endl;
}

void parseCommandline(int argc, char *argv[], App &app) {
    int idx = 1;

    while (idx < argc) {
        if (0 == std::string("-n").compare(argv[idx])) {
            app._pointNum = atoi(argv[idx + 1]);
            ++idx;
        } else if (0 == std::string("-c").compare(argv[idx])) {
            app._constraintNum = atoi(argv[idx + 1]);
            ++idx;
        } else if (0 == std::string("-r").compare(argv[idx])) {
            app._runNum = atoi(argv[idx + 1]);
            ++idx;
        } else if (0 == std::string("-seed").compare(argv[idx])) {
            app._seed = atoi(argv[idx + 1]);
            ++idx;
        } else if (0 == std::string("-check").compare(argv[idx])) {
            app._doCheck = true;
        } else if (0 == std::string("-d").compare(argv[idx])) {
            const int distVal = atoi(argv[idx + 1]);
            app._dist = (Distribution) distVal;

            ++idx;
        } else if (0 == std::string("-inFile").compare(argv[idx])) {
            app._inFile = true;
            app._inFilename = std::string(argv[idx + 1]);

            ++idx;
        } else if (0 == std::string("-insAll").compare(argv[idx])) {
            app._input.insAll = true;
        } else if (0 == std::string("-noSort").compare(argv[idx])) {
            app._input.noSort = true;
        } else if (0 == std::string("-noReorder").compare(argv[idx])) {
            app._input.noReorder = true;
        } else if (0 == std::string("-profiling").compare(argv[idx])) {
            app._input.profLevel = ProfDetail;
        } else if (0 == std::string("-diag").compare(argv[idx])) {
            app._input.profLevel = ProfDiag;
        } else if (0 == std::string("-debug").compare(argv[idx])) {
            app._input.profLevel = ProfDebug;
        } else {
            std::cout << "Error in input argument: " << argv[idx] << "\n\n";
            std::cout << "Syntax: GDelFlipping [-n PointNum][-r RunNum][-seed SeedNum][-d DistNum]\n";
            std::cout << "                     [-inFile FileName][-insAll][-noSort][-noReorder][-check]\n";
            std::cout << "                     [-profiling][-diag][-debug]\n";
            std::cout << "Dist: 0-Uniform 1-Gaussian 2-Disk 3-Circle 4-Grid 5-Ellipse 6-TwoLines\n";

            exit(1);
        }

        ++idx;
    }
}

int main(int argc, char *argv[]) {
    App app;

    parseCommandline(argc, argv, app);

    Visualizer::instance()->disable();

    // Run test
    app.run();

    summarize(app);

    return 0;
}
