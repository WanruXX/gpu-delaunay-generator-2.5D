#include "../inc/TriangulationHandler.h"
#include "../inc/json.h"
#include <unistd.h>
#include <yaml-cpp/yaml.h>

TriangulationHandler::TriangulationHandler(const char *InputYAMLFile)
{
    CudaSafeCall(cudaSetDevice(cutGetMaxGflopsDeviceId()));

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
    input.insAll    = config["InsertAll"].as<bool>();
    input.noSort    = config["NoSortPoint"].as<bool>();
    input.noReorder = config["NoReorder"].as<bool>();

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

    outputResult   = config["OutputTriangles"].as<bool>();
    OutputFilename = config["OutputTrianglePath"].as<std::string>();
}

void TriangulationHandler::reset()
{
    TriHVec().swap(output.triVec);
    TriOppHVec().swap(output.triOppVec);
}

void TriangulationHandler::run()
{
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
    std::ofstream outputTri(OutputFilename);
    if (outputTri.is_open())
    {
        std::size_t found = OutputFilename.find_last_of('.');
        std::string extension = OutputFilename.substr(found + 1, OutputFilename.size() - found);
        if ( extension == "obj")
        {
            saveToObj(outputTri);
        }
        else if(extension == "geojson")
        {
            saveToGeojson(outputTri);
        }
        else{
            std::cerr << "Can't identify the saving file's extension!" << std::endl;
        }
        outputTri.close();
    }
    else
    {
        std::cerr << "Delaunay triangulation saving path " << OutputFilename << " is not valid! will not save..."
                  << std::endl;
    }
}

void TriangulationHandler::saveToGeojson(std::ofstream &outputTri) const{
    nlohmann::json JsonFile;
    JsonFile["type"]                      = "FeatureCollection";
    JsonFile["name"]                      = "left_4_edge_polygon";
    JsonFile["crs"]["type"]               = "name";
    JsonFile["crs"]["properties"]["name"] = "urn:ogc:def:crs:EPSG::32601";
    JsonFile["features"]                  = nlohmann::json::array();
    for (auto &tri : output.triVec)
    {
        nlohmann::json Coor = nlohmann::json::array();
        Coor.push_back({input.pointVec[tri._v[0]]._p[0],
                        input.pointVec[tri._v[0]]._p[1],
                        input.pointVec[tri._v[0]]._p[2]});
        Coor.push_back({input.pointVec[tri._v[1]]._p[0],
                        input.pointVec[tri._v[1]]._p[1],
                        input.pointVec[tri._v[1]]._p[2]});
        Coor.push_back({input.pointVec[tri._v[2]]._p[0],
                        input.pointVec[tri._v[2]]._p[1],
                        input.pointVec[tri._v[2]]._p[2]});
        nlohmann::json CoorWrapper = nlohmann::json::array();
        CoorWrapper.push_back(Coor);
        nlohmann::json TriangleObject;
        TriangleObject["type"]       = "Feature";
        TriangleObject["properties"] = {{"v0", tri._v[0]}, {"v1", tri._v[1]}, {"v2", tri._v[2]}};
        TriangleObject["geometry"]   = {{"type", "Polygon"}, {"coordinates", CoorWrapper}};
        JsonFile["features"].push_back(TriangleObject);
    }
    outputTri << JsonFile << std::endl;
}

void TriangulationHandler::saveToObj(std::ofstream &outputTri) const{
    outputTri << std::setprecision(12);
    double colors[5][3] = {
        {0.8, 0.2, 0.2},  // Red
        {0.9, 0.6, 0.2},  // Orange
        {0.2, 0.8, 0.2},  // Green
        {0.2, 0.6, 0.9},  // Sky Blue
        {0.6, 0.2, 0.8}   // Purple
    };
    int i = 0;
    for (const auto &pt : input.pointVec)
    {
        double *color = colors[i % 5];
        outputTri << "v " << pt._p[0] << " " << pt._p[1] << " " << pt._p[2] << " " << color[0] << " "
                  << color[1] << " " << color[2] << std::endl;
        ++i;
    }
    for (auto &tri : output.triVec)
    {
        outputTri << "f " << tri._v[0] + 1 << " " << tri._v[1] + 1 << " " << tri._v[2] + 1 << std::endl;
    }
}

bool TriangulationHandler::checkInside(Tri &t, Point p) const
{
    // Create a point at infinity, y is same as point p
    line exline = {p, {p._p[0] + 320, p._p[1], p._p[2]}};
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