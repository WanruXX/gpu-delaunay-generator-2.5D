#ifndef DELAUNAY_GENERATOR_INPUTCREATOR_H
#define DELAUNAY_GENERATOR_INPUTCREATOR_H

#include "CommonTypes.h"
#include "IOType.h"
#include "RandGen.h"

enum Distribution
{
    UniformDistribution,
    GaussianDistribution,
    DiskDistribution,
    ThinCircleDistribution,
    CircleDistribution,
    GridDistribution,
    EllipseDistribution,
    TwoLineDistribution
};

struct InputGeneratorOption
{
    bool         inputFromFile = false;
    std::string  inputFilename;
    bool         inputConstraint = false;
    std::string  inputConstraintFilename;
    int          pointNum     = 1000;
    Distribution distribution = UniformDistribution;
    int          seed         = 76213898;
    bool         saveToFile   = false;
    std::string  saveFilename;

    void setDistributionFromStr(const std::string &distributionStr);
};

class InputGenerator
{
  private:
    RandGen                     randGen;
    const InputGeneratorOption &option;
    Input                      &input;

    void randCirclePoint(double &x, double &y);

    void makePoints();

    void makePointsUniform();

    void makePointsGaussian();

    void makePointsDisk();

    void makePointsThinCircle();

    void makePointsCircle();

    void makePointsGrid();

    void makePointsEllipse();

    void makePointsTwoLine();

    void readPoints();

    void readConstraints();

  public:
    InputGenerator() = delete;
    explicit InputGenerator(const InputGeneratorOption &InputPara, Input &Input);
    void generateInput();
};

#endif //DELAUNAY_GENERATOR_INPUTCREATOR_H