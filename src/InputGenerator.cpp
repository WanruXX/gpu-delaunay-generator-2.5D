#include "../inc/InputGenerator.h"
#include "../inc/HashFunctors.h"
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#ifdef WITH_PCL
#include <pcl/io/pcd_io.h>
#endif

namespace
{
const std::unordered_map<std::string, Distribution> distributionMap{{"Uniform", UniformDistribution},
                                                                    {"Gaussian", GaussianDistribution},
                                                                    {"Disk", DiskDistribution},
                                                                    {"ThinCircle", ThinCircleDistribution},
                                                                    {"Circle", CircleDistribution},
                                                                    {"Grid", GridDistribution},
                                                                    {"Ellipse", EllipseDistribution},
                                                                    {"TwoLines", TwoLineDistribution}};
}

InputGenerator::InputGenerator(const InputGeneratorOption &InputPara, Input &Input) : option(InputPara), input(Input)
{
}

void InputGeneratorOption::setDistributionFromStr(const std::string &distributionStr)
{
    if (distributionMap.find(distributionStr) == distributionMap.end())
    {
        throw std::invalid_argument("Input: cannot find the distribution specified!");
    }
    distribution = distributionMap.at(distributionStr);
}

void InputGenerator::randCirclePoint(double &x, double &y)
{
    double a = randGen.getNext() * M_PI * 2;
    x        = 0.45 * cos(a);
    y        = 0.45 * sin(a);
}

void InputGenerator::generateInput()
{
    if (option.inputFromFile)
    {
        readPoints();
        if (option.inputConstraint)
        {
            readConstraints();
        }
    }
    else
    {
        makePoints();
        input.removeDuplicates();
    }

    if (option.saveToFile)
    {
        std::ofstream outputPoint(option.saveFilename);
        if (outputPoint.is_open())
        {
            outputPoint << std::setprecision(12);
            for (const auto &pt : input.pointVec)
            {
                outputPoint << pt._p[0] << " " << pt._p[1] << std::endl;
            }
            outputPoint.close();
        }
        else
        {
            std::cerr << "Point saving path " << option.saveFilename << "is not valid! will not save..." << std::endl;
        }

        if (option.inputConstraint)
        {
            std::ofstream outputConstraint(option.saveFilename.substr(0, option.saveFilename.size() - 4) +
                                           "_constraints.txt");
            if (outputConstraint.is_open())
            {
                outputConstraint << std::setprecision(12);
                for (const auto &edge : input.constraintVec)
                {
                    outputConstraint << edge._v[0] << " " << edge._v[1] << std::endl;
                }
                outputConstraint.close();
            }
        }
    }

    std::cout << "Number of input points:      " << input.pointVec.size() << std::endl;
    std::cout << "Number of input constraints: " << input.constraintVec.size() << std::endl;
}

void InputGenerator::makePoints()
{
    input.pointVec.reserve(option.pointNum);
    randGen.init(option.seed, 0.0, 1.0);

    switch (option.distribution)
    {
    case UniformDistribution:
        makePointsUniform();
        break;
    case GaussianDistribution:
        makePointsGaussian();
        break;
    case DiskDistribution:
        makePointsDisk();
        break;
    case ThinCircleDistribution:
        makePointsThinCircle();
        break;
    case CircleDistribution:
        makePointsCircle();
        break;
    case GridDistribution:
        makePointsGrid();
        break;
    case EllipseDistribution:
        makePointsEllipse();
        break;
    case TwoLineDistribution:
        makePointsTwoLine();
        break;
    default:
        makePointsUniform();
        break;
    }
}

void InputGenerator::makePointsUniform()
{
    std::unordered_set<Point2D, Point2DHash> pointSet;
    Point2D                                  p;
    while (pointSet.size() < option.pointNum)
    {
        p._p[0] = randGen.getNext();
        p._p[1] = randGen.getNext();
        if (pointSet.find(p) == pointSet.end())
        {
            pointSet.insert(p);
            input.pointVec.push_back(p);
        }
    }
}

void InputGenerator::makePointsGaussian()
{
    std::unordered_set<Point2D, Point2DHash> pointSet;
    Point2D                                  p;
    while (pointSet.size() < option.pointNum)
    {
        randGen.nextGaussian(p._p[0], p._p[1]);
        if (pointSet.find(p) == pointSet.end())
        {
            pointSet.insert(p);
            input.pointVec.push_back(p);
        }
    }
}

void InputGenerator::makePointsDisk()
{
    std::unordered_set<Point2D, Point2DHash> pointSet;
    Point2D                                  p;
    while (pointSet.size() < option.pointNum)
    {
        double d;
        do
        {
            p._p[0] = randGen.getNext() - 0.5;
            p._p[1] = randGen.getNext() - 0.5;
            d       = p._p[0] * p._p[0] + p._p[1] * p._p[1];
        } while (d > 0.45 * 0.45);
        p._p[0] += 0.5;
        p._p[1] += 0.5;
        if (pointSet.find(p) == pointSet.end())
        {
            pointSet.insert(p);
            input.pointVec.push_back(p);
        }
    }
}

void InputGenerator::makePointsThinCircle()
{
    std::unordered_set<Point2D, Point2DHash> pointSet;
    Point2D                                  p;
    while (pointSet.size() < option.pointNum)
    {
        double d = randGen.getNext() * 0.001;
        double a = randGen.getNext() * 3.141592654 * 2;
        p._p[0]  = (0.45 + d) * cos(a) + 0.5;
        p._p[1]  = (0.45 + d) * sin(a) + 0.5;
        if (pointSet.find(p) == pointSet.end())
        {
            pointSet.insert(p);
            input.pointVec.push_back(p);
        }
    }
}

void InputGenerator::makePointsCircle()
{
    std::unordered_set<Point2D, Point2DHash> pointSet;
    Point2D                                  p;
    while (pointSet.size() < option.pointNum)
    {
        randCirclePoint(p._p[0], p._p[1]);
        p._p[0] += 0.5;
        p._p[1] += 0.5;
        if (pointSet.find(p) == pointSet.end())
        {
            pointSet.insert(p);
            input.pointVec.push_back(p);
        }
    }
}

void InputGenerator::makePointsGrid()
{
    std::unordered_set<Point2D, Point2DHash> pointSet;
    Point2D                                  p;
    while (pointSet.size() < option.pointNum)
    {
        for (double &pp : p._p)
        {
            const double val  = randGen.getNext() * 8192;
            const double frac = val - floor(val);
            pp                = (frac < 0.5f) ? floor(val) : ceil(val);
            pp /= 8192;
        }
        if (pointSet.find(p) == pointSet.end())
        {
            pointSet.insert(p);
            input.pointVec.push_back(p);
        }
    }
}

void InputGenerator::makePointsEllipse()
{
    std::unordered_set<Point2D, Point2DHash> pointSet;
    Point2D                                  p;
    while (pointSet.size() < option.pointNum)
    {
        randCirclePoint(p._p[0], p._p[1]);
        p._p[0] = p._p[0] * 1.0 / 3.0 + 0.5;
        p._p[1] = p._p[1] * 2.0 / 3.0 + 0.5;
        if (pointSet.find(p) == pointSet.end())
        {
            pointSet.insert(p);
            input.pointVec.push_back(p);
        }
    }
}

void InputGenerator::makePointsTwoLine()
{
    std::unordered_set<Point2D, Point2DHash> pointSet;
    Point2D                                  p;

    const Point2D L[2][2] = {{{0.0, 0.0}, {0.3, 0.5}}, {{0.7, 0.5}, {1.0, 1.0}}};
    while (pointSet.size() < option.pointNum)
    {
        int    l = (randGen.getNext() < 0.5) ? 0 : 1;
        double t = randGen.getNext();
        p._p[0]  = (L[l][1]._p[0] - L[l][0]._p[0]) * t + L[l][0]._p[0];
        p._p[0]  = (L[l][1]._p[1] - L[l][0]._p[1]) * t + L[l][0]._p[1];
        if (pointSet.find(p) == pointSet.end())
        {
            pointSet.insert(p);
            input.pointVec.push_back(p);
        }
    }
}

void InputGenerator::readPoints()
{
#ifdef WITH_PCL
    if (inFilename.substr(inFilename.size() - 4, 4) == ".pcd")
    {
        pcl::PointCloud<POINT_TYPE> inputPointCloud;
        pcl::io::loadPCDFile(inFilename, inputPointCloud);
        for (auto &pt : inputPointCloud)
        {
            input.pointVec.push_back({pt.x, pt.y});
        }
        return input.pointVec;
    }
#endif
    std::ifstream inFile(option.inputFilename);
    std::string   LineData;
    Point2D       pt;
    while (std::getline(inFile, LineData))
    {
        std::stringstream ss(LineData);
        ss >> pt._p[0] >> pt._p[1];
        input.pointVec.push_back(pt);
    }
    inFile.close();
}

void InputGenerator::readConstraints()
{
    std::ifstream inFile(option.inputConstraintFilename);
    std::string   LineData;
    Edge          edge;
    while (std::getline(inFile, LineData))
    {
        std::stringstream ss(LineData);
        ss >> edge._v[0] >> edge._v[1];
        input.constraintVec.push_back(edge);
    }
    inFile.close();
}