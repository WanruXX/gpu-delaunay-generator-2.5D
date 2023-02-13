#include "../inc/InputCreator.h"
#include "../inc/PointType.h"
#include <cmath>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>

void InputCreator::randCirclePoint(double &x, double &y)
{
    double a = randGen.getNext() * M_PI * 2;
    x        = 0.45 * cos(a);
    y        = 0.45 * sin(a);
}

void InputCreator::createPoints(const InputCreatorPara &InputPara, Point2HVec &pointVec, SegmentHVec &constraintVec)
{
    if (InputPara._inFile)
    {
        readPoints(InputPara._inFilename);
        if (InputPara._constraintNum == 0)
        {
            readConstraints(InputPara._inConstraintFilename);
        }
        std::cout << "Number of input points:      " << inPointVec.size() << std::endl;
        std::cout << "Number of input constraints: " << inConstraintVec.size() << std::endl;

        // Remove duplicates
        HashPoint2 hashPoint2;
        PointTable pointSet(static_cast<int>(inPointVec.size()), hashPoint2);
        IntHVec    pointMap(inPointVec.size());
        // Iterate input points
        for (size_t ip = 0; ip < inPointVec.size(); ++ip)
        {
            Point2 inPt = inPointVec[ip];
            int    ptIdx;
            // Check if point unique
            if (!pointSet.get(inPt, &ptIdx))
            {
                pointVec.push_back(inPt);
                ptIdx = static_cast<int>(pointVec.size()) - 1;
                pointSet.insert(inPt, ptIdx);
            }
            pointMap[ip] = ptIdx;
        }
        const auto dupCount = inPointVec.size() - pointVec.size();
        if (dupCount > 0)
        {
            std::cout << dupCount << " duplicate points in input file!" << std::endl;
        }
        // Iterate input constraints
        for (size_t i = 0; i < inConstraintVec.size(); ++i)
        {
            const Segment inC  = inConstraintVec[i];
            const Segment newC = {pointMap[inC._v[0]], pointMap[inC._v[1]]};
            if (newC._v[0] != newC._v[1] && constraintVec.size() < inConstraintVec.size())
            {
                constraintVec.push_back(newC);
            }
        }
        const auto dupConstraint = inConstraintVec.size() - constraintVec.size();
        if (dupConstraint > 0)
            std::cout << dupConstraint << " degenerate or ignored constraints in input file!" << std::endl;
    }
    else
    {
        makePoints(InputPara._pointNum, InputPara._dist, pointVec, InputPara._seed);
        if (InputPara._saveToFile)
        {
            std::ofstream OutputFile(InputPara._savePath);
            if (OutputFile.is_open())
            {
                OutputFile << std::setprecision(12);
                for (const auto &pt : inPointVec)
                {
                    OutputFile << pt._p[0] << " " << pt._p[1] << std::endl;
                }
                OutputFile.close();
            }
            else
            {
                std::cerr << InputPara._savePath << " is not a valid path!" << std::endl;
            }
        }
    }
}

void InputCreator::createPoints(const InputCreatorPara &InputPara, const pcl::PointCloud<POINT_TYPE>::Ptr &InputPC, Point2HVec &pointVec, SegmentHVec &constraintVec)
{
    if (InputPara._inFile)
    {
        readPoints(InputPara._inFilename, InputPC);
        if (InputPara._constraintNum == 0)
        {
            readConstraints(InputPara._inConstraintFilename);
        }
        std::cout << "Number of input points:      " << inPointVec.size() << std::endl;
        std::cout << "Number of input constraints: " << inConstraintVec.size() << std::endl;

        // Remove duplicates
        HashPoint2 hashPoint2;
        PointTable pointSet(static_cast<int>(inPointVec.size()), hashPoint2);
        IntHVec    pointMap(inPointVec.size());
        pcl::PointIndices::Ptr FilterIdx(new pcl::PointIndices);
        // Iterate input points
        for (size_t ip = 0; ip < inPointVec.size(); ++ip)
        {
            Point2 inPt = inPointVec[ip];
            int    ptIdx;
            // Check if point unique
            if (!pointSet.get(inPt, &ptIdx))
            {
                pointVec.push_back(inPt);
                ptIdx = static_cast<int>(pointVec.size()) - 1;
                pointSet.insert(inPt, ptIdx);
                FilterIdx->indices.push_back(static_cast<int>(ip));
            }
            pointMap[ip] = ptIdx;
        }
        pcl::ExtractIndices<POINT_TYPE> extract;
        extract.setInputCloud(InputPC);
        extract.setIndices(FilterIdx);
        extract.filter(*InputPC);
        const auto dupCount = inPointVec.size() - pointVec.size();
        if (dupCount > 0)
        {
            std::cout << dupCount << " duplicate points in input file!" << std::endl;
        }
        // Iterate input constraints
        for (size_t i = 0; i < inConstraintVec.size(); ++i)
        {
            const Segment inC  = inConstraintVec[i];
            const Segment newC = {pointMap[inC._v[0]], pointMap[inC._v[1]]};
            if (newC._v[0] != newC._v[1] && constraintVec.size() < inConstraintVec.size())
            {
                constraintVec.push_back(newC);
            }
        }
        const auto dupConstraint = inConstraintVec.size() - constraintVec.size();
        if (dupConstraint > 0)
            std::cout << dupConstraint << " degenerate or ignored constraints in input file!" << std::endl;
    }
    else
    {
        makePoints(InputPara._pointNum, InputPara._dist, pointVec, InputPara._seed);
        if (InputPara._saveToFile)
        {
            std::ofstream OutputFile(InputPara._savePath);
            if (OutputFile.is_open())
            {
                OutputFile << std::setprecision(12);
                for (const auto &pt : inPointVec)
                {
                    OutputFile << pt._p[0] << " " << pt._p[1] << std::endl;
                }
                OutputFile.close();
            }
            else
            {
                std::cerr << InputPara._savePath << " is not a valid path!" << std::endl;
            }
        }
    }
}

void InputCreator::makePoints(int pointNum, Distribution dist, Point2HVec &pointVec, int seed)
{
    pointVec.reserve(pointNum);
    HashPoint2 hashPoint2;
    PointTable pointSet(pointNum, hashPoint2);

    // Initialize seed
    randGen.init(seed, 0.0, 1.0);

    // Generate rest of points
    double x = 0.0;
    double y = 0.0;
    Point2 p;
    // Generate rest of points randomly
    for (int i = 0; i < pointNum; ++i)
    {
        do
        {
            switch (dist)
            {
            case UniformDistribution:
            {
                x = randGen.getNext();
                y = randGen.getNext();
            }
            break;

            case GaussianDistribution:
            {
                randGen.nextGaussian(x, y);
            }
            break;

            case DiskDistribution:
            {
                double d;

                do
                {
                    x = randGen.getNext() - 0.5;
                    y = randGen.getNext() - 0.5;

                    d = x * x + y * y;

                } while (d > 0.45 * 0.45);

                x += 0.5;
                y += 0.5;
            }
            break;

            case ThinCircleDistribution:
            {
                double d, a;

                d = randGen.getNext() * 0.001;
                a = randGen.getNext() * 3.141592654 * 2;

                x = (0.45 + d) * cos(a);
                y = (0.45 + d) * sin(a);

                x += 0.5;
                y += 0.5;
            }
            break;

            case CircleDistribution:
            {
                randCirclePoint(x, y);

                x += 0.5;
                y += 0.5;
            }
            break;

            case GridDistribution:
            {
                double v[2];
                for (double &vv : v)
                {
                    const double val  = randGen.getNext() * GridSize;
                    const double frac = val - floor(val);
                    vv                = (frac < 0.5f) ? floor(val) : ceil(val);
                    vv /= GridSize;
                }

                x = v[0];
                y = v[1];
            }
            break;

            case EllipseDistribution:
            {
                randCirclePoint(x, y);

                x = x * 1.0 / 3.0;
                y = y * 2.0 / 3.0;

                x += 0.5;
                y += 0.5;
            }
            break;

            case TwoLineDistribution:
            {
                const Point2 L[2][2] = {{{0.0, 0.0}, {0.3, 0.5}}, {{0.7, 0.5}, {1.0, 1.0}}};

                const int    l = (randGen.getNext() < 0.5) ? 0 : 1;
                const double t = randGen.getNext(); // [ 0, 1 ]

                x = (L[l][1]._p[0] - L[l][0]._p[0]) * t + L[l][0]._p[0];
                y = (L[l][1]._p[1] - L[l][0]._p[1]) * t + L[l][0]._p[1];
            }
            break;
            }

            p._p[0] = x;
            p._p[1] = y;

        } while (pointSet.get(p, nullptr));

        pointVec.push_back(p);
        pointSet.insert(p, static_cast<int>(pointVec.size() - 1));
    }
}

void InputCreator::readPoints(const std::string &inFilename)
{
    if (inFilename.substr(inFilename.size() - 4, 4) == ".pcd")
    {
        pcl::PointCloud<POINT_TYPE> pc;
        pcl::io::loadPCDFile(inFilename, pc);
        for (auto &pt : pc)
        {
            inPointVec.push_back(Point2(pt.x, pt.y));
        }
    }
    else
    {
        std::ifstream inFile(inFilename);
        std::string   LineData;
        Point2        pt;
        while (std::getline(inFile, LineData))
        {
            std::stringstream ss(LineData);
            ss >> pt._p[0] >> pt._p[1];
            inPointVec.push_back(pt);
        }
        inFile.close();
    }
}

void InputCreator::readPoints(const std::string &inFilename, const pcl::PointCloud<POINT_TYPE>::Ptr &InputPC)
{
    if (inFilename.substr(inFilename.size() - 4, 4) == ".pcd")
    {
        pcl::io::loadPCDFile(inFilename, *InputPC);
        for (auto &pt : *InputPC)
        {
            inPointVec.push_back(Point2(pt.x, pt.y));
        }
    }
    else
    {
        std::ifstream inFile(inFilename);
        std::string   LineData;
        Point2        pt;
        while (std::getline(inFile, LineData))
        {
            std::stringstream ss(LineData);
            ss >> pt._p[0] >> pt._p[1];
            inPointVec.push_back(pt);
        }
        inFile.close();
    }
}

void InputCreator::readConstraints(const std::string &inFilename)
{
    std::ifstream inFile(inFilename);
    std::string   LineData;
    Segment       seg;
    while (std::getline(inFile, LineData))
    {
        std::stringstream ss(LineData);
        ss >> seg._v[0] >> seg._v[1];
        inConstraintVec.push_back(seg);
    }
    inFile.close();
}