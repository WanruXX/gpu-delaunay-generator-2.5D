#include "../inc/TriangulationHandler.h"
#include "../inc/json.h"

#ifdef WITH_PCL
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#endif

#include <unistd.h>
#include <yaml-cpp/yaml.h>

TriangulationHandler::TriangulationHandler(const char *InputYAMLFile)
{
    YAML::Node config = YAML::LoadFile(InputYAMLFile);

    runNum  = config["RunNum"].as<int>();
    doCheck = config["DoCheck"].as<bool>();

    InputGeneratorOption inputGeneratorOption;
    inputGeneratorOption.inputFromFile = config["InputFromFile"].as<bool>();
    if (inputGeneratorOption.inputFromFile)
    {
        inputGeneratorOption.inputFilename = config["InputPointCloudFile"].as<std::string>();
        if (access(inputGeneratorOption.inputFilename.c_str(), F_OK) == -1)
        {
            throw std::invalid_argument("Input point cloud file doesn't exist!");
        }
        inputGeneratorOption.inputConstraintFilename = config["InputConstraintFile"].as<std::string>();
        if (access(inputGeneratorOption.inputConstraintFilename.c_str(), F_OK) == -1)
        {
            std::cerr << "Input constraints file " << inputGeneratorOption.inputConstraintFilename
                      << " doesn't exist, not using constraints..." << std::endl;
        }
        else
        {
            inputGeneratorOption.inputConstraint = true;
        }
    }
    else
    {
        inputGeneratorOption.pointNum = config["PointNum"].as<int>();
        inputGeneratorOption.setDistributionFromStr(config["DistributionType"].as<std::string>());
        inputGeneratorOption.seed = config["Seed"].as<int>();
    }
    inputGeneratorOption.saveToFile   = config["SaveToFile"].as<bool>();
    inputGeneratorOption.saveFilename = config["SavePath"].as<std::string>();

    auto           timer = (double)clock();
    InputGenerator inputGenerator(inputGeneratorOption, input);
    inputGenerator.generateInput();
    std::cout << "Point generating time: " << ((double)clock() - timer) / CLOCKS_PER_SEC << std::endl;
    input.insAll               = config["InsertAll"].as<bool>();
    input.noSort               = config["NoSortPoint"].as<bool>();
    input.noReorder            = config["NoReorder"].as<bool>();
    std::string InputProfLevel = config["ProfLevel"].as<std::string>();
    if (InputProfLevel == "Detail")
    {
        input.profLevel = ProfDetail;
    }
    else if (InputProfLevel == "Diag")
    {
        input.profLevel = ProfDiag;
    }
    else if (InputProfLevel == "Debug")
    {
        input.profLevel = ProfDebug;
    }
    else
    {
        input.profLevel = ProfDefault;
    }

    outputResult   = config["OutputResult"].as<bool>();
    outTriFilename = config["OutputMeshPath"].as<std::string>();
    if (access(inputGeneratorOption.saveFilename.c_str(), F_OK) == -1)
    {
        std::cerr << "Saving path for generated points is not valid! will not save..." << std::endl;
        inputGeneratorOption.saveToFile = false;
    }
}

void TriangulationHandler::reset()
{
    TriHVec().swap(output.triVec);
    TriOppHVec().swap(output.triOppVec);
    cudaDeviceReset();
}

void TriangulationHandler::run()
{
    // Pick the best CUDA device
    const int deviceIdx = cutGetMaxGflopsDeviceId();
    CudaSafeCall(cudaSetDevice(deviceIdx));
    CudaSafeCall(cudaDeviceReset());

    GpuDel gpuDel;
    for (int i = 0; i < runNum; ++i)
    {
        reset();
        gpuDel.compute(input, output);
        std::cout << "Run " << i << " ---> gpu usage time (ms): " << output.stats.totalTime << " ("
                  << output.stats.initTime << ", " << output.stats.splitTime << ", " << output.stats.flipTime << ", "
                  << output.stats.relocateTime << ", " << output.stats.sortTime << ", " << output.stats.constraintTime
                  << ", " << output.stats.outTime << ")" << std::endl;
        statSum.accumulate(output.stats);
        if (doCheck)
        {
            DelaunayChecker checker(input, output);
            std::cout << "\n*** Check ***\n";
            checker.checkEuler();
            checker.checkOrientation();
            checker.checkAdjacency();
            checker.checkConstraints();
            checker.checkDelaunay();
        }
    }
    statSum.average(runNum);

    if (outputResult)
    {
        saveResultsToFile();
    }

    std::cout << std::endl;
    std::cout << "---- SUMMARY ----" << std::endl;
    std::cout << std::endl;
    std::cout << "PointNum       " << input.pointVec.size() << std::endl;
    std::cout << "Sort           " << (input.noSort ? "no" : "yes") << std::endl;
    std::cout << "Reorder        " << (input.noReorder ? "no" : "yes") << std::endl;
    std::cout << "Insert mode    " << (input.insAll ? "InsAll" : "InsFlip") << std::endl;
    std::cout << std::endl;
    std::cout << std::fixed << std::right << std::setprecision(2);
    std::cout << "TotalTime (ms) " << std::setw(10) << statSum.totalTime << std::endl;
    std::cout << "InitTime       " << std::setw(10) << statSum.initTime << std::endl;
    std::cout << "SplitTime      " << std::setw(10) << statSum.splitTime << std::endl;
    std::cout << "FlipTime       " << std::setw(10) << statSum.flipTime << std::endl;
    std::cout << "RelocateTime   " << std::setw(10) << statSum.relocateTime << std::endl;
    std::cout << "SortTime       " << std::setw(10) << statSum.sortTime << std::endl;
    std::cout << "ConstraintTime " << std::setw(10) << statSum.constraintTime << std::endl;
    std::cout << "OutTime        " << std::setw(10) << statSum.outTime << std::endl;
    std::cout << std::endl;
}

void TriangulationHandler::saveResultsToFile()
{
    std::ofstream outputTri(outTriFilename);
    if (outputTri.is_open())
    {
        nlohmann::json JsonFile;
        JsonFile["type"]                      = "FeatureCollection";
        JsonFile["name"]                      = "left_4_edge_polygon";
        JsonFile["crs"]["type"]               = "name";
        JsonFile["crs"]["properties"]["name"] = "urn:ogc:def:crs:EPSG::32601";
        JsonFile["features"]                  = nlohmann::json::array();
        for (auto &tri : output.triVec)
        {
            nlohmann::json Coor = nlohmann::json::array();
            Coor.push_back({static_cast<double>(input.pointVec[tri._v[0]]._p[0]),
                            static_cast<double>(input.pointVec[tri._v[0]]._p[1])});
            Coor.push_back({static_cast<double>(input.pointVec[tri._v[1]]._p[0]),
                            static_cast<double>(input.pointVec[tri._v[1]]._p[1])});
            Coor.push_back({static_cast<double>(input.pointVec[tri._v[2]]._p[0]),
                            static_cast<double>(input.pointVec[tri._v[2]]._p[1])});
            nlohmann::json CoorWrapper = nlohmann::json::array();
            CoorWrapper.push_back(Coor);
            nlohmann::json TriangleObject;
            TriangleObject["type"]       = "Feature";
            TriangleObject["properties"] = {{"v0", tri._v[0]}, {"v1", tri._v[1]}, {"v2", tri._v[2]}};
            TriangleObject["geometry"]   = {{"type", "Polygon"}, {"coordinates", CoorWrapper}};
            JsonFile["features"].push_back(TriangleObject);
        }
        outputTri << JsonFile << std::endl;
        outputTri.close();
    }
    else
    {
        std::cerr << "Delaunay triangulation saving path " << outTriFilename << " is not valid! will not save..."
                  << std::endl;
    }
}

bool TriangulationHandler::checkInside(Tri &t, Point2D p) const
{
    // Create a point at infinity, y is same as point p
    line exline = {p, {p._p[0] + 320, p._p[1]}};
    int  count  = 0;
    for (auto i : TriSeg)
    {
        line side = {input.pointVec[t._v[i[0]]], input.pointVec[t._v[i[1]]]};
        if (isIntersect(side, exline))
        {

            // If side intersects exline
            if (direction(side.p1, p, side.p2) == 0)
                return onLine(side, p);
            count++;
        }
    }
    // When count is odd
    return count & 1;
}

#ifdef WITH_PCL
double TriangulationHandler::getupwards(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3)
{
    Eigen::Vector3d Upward(0, 0, 1);
    auto            Norm = getTriNormal(pt1, pt2, pt3);
    return std::acos(Norm.dot(Upward)) * 180 / M_PI;
}

Eigen::Vector3d
TriangulationHandler::getTriNormal(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3)
{
    //    Eigen::Vector3d pt1pt2(pt2.x - pt1.x, pt2.y - pt1.y, pt2.z - pt1.z);
    //    Eigen::Vector3d pt2pt3(pt3.x - pt2.x, pt3.y - pt2.y, pt3.z - pt2.z);
    Eigen::Vector3d pt1pt2(pt2.x - pt1.x, pt2.y - pt1.y, pt2.classflags - pt1.classflags);
    Eigen::Vector3d pt2pt3(pt3.x - pt2.x, pt3.y - pt2.y, pt3.classflags - pt2.classflags);
    auto            Normal = pt1pt2.cross(pt2pt3);
    auto            Norm   = Normal(2) > 0 ? Normal : -Normal;
    //    double          c0     = pt1pt2(1) * pt2pt3(2) - pt1pt2(2) * pt2pt3(1);
    //    double          c1     = pt1pt2(2) * pt2pt3(0) - pt1pt2(0) * pt2pt3(2);
    //    double          c2     = pt1pt2(0) * pt2pt3(1) - pt1pt2(1) * pt2pt3(0);
    //    std::cout << std::setprecision(10);
    //    std::cout << pt1pt2(0) << " " << pt1pt2(1) << " " << pt1pt2(2) << std::endl;
    //    std::cout << pt2pt3(0) << " " << pt2pt3(1) << " " << pt2pt3(2) << std::endl;
    //    std::cout << Norm(0) << " " << Norm(1) << " " << Norm(2) << std::endl;
    //    std::cout << c0 << " " << c1 << " " << c2 << std::endl;
    Norm.normalize();
    return Norm;
}

Eigen::Vector3d TriangulationHandler::getTriNormal(const Tri &t) const
{
    auto pt1 = InputPointCloud[t._v[0]];
    auto pt2 = InputPointCloud[t._v[1]];
    auto pt3 = InputPointCloud[t._v[2]];
    return getTriNormal(pt1, pt2, pt3);
}

bool TriangulationHandler::hasValidEdge(const tTrimblePoint &pt1, const tTrimblePoint &pt2, const tTrimblePoint &pt3)
{
    return (pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y) < 1 &&
           (pt3.x - pt2.x) * (pt3.x - pt2.x) + (pt3.y - pt2.y) * (pt3.y - pt2.y) < 1 &&
           (pt1.x - pt3.x) * (pt1.x - pt3.x) + (pt1.y - pt3.y) * (pt1.y - pt3.y) < 1;
}
#endif