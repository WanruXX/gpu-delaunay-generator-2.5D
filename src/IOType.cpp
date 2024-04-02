#include "../inc/IOType.h"
#include "../inc/HashFunctors.h"
#include <unordered_set>

void Input::removeDuplicates(){
    Point2DHVec  oldPointVec = std::move(pointVec);
    EdgeHVec     oldConstraintVec = std::move(constraintVec);

    std::unordered_set<Point2D, Point2DHash> pointSet;
    std::vector<int>    pointMap(oldPointVec.size());
    for (std::size_t i = 0; i < oldPointVec.size(); ++i)
    {
        const auto &pt = oldPointVec[i];
        if (pointSet.find(pt) == pointSet.end())
        {
            pointSet.insert(pt);
            pointVec.push_back(pt);
        }
        else
        {
            std::cout << " Duplicate point found at [" << i << "], will remove from indexing!" << std::endl;
        }
        pointMap[i] = static_cast<int>(pointVec.size()) - 1;;
    }

    std::unordered_set<Edge, EdgeHash, EdgeEqual> edgeSet;
    for (auto &con : oldConstraintVec)
    {
        Edge edge = {pointMap[con._v[0]], pointMap[con._v[1]]};
        if (edge._v[0] != edge._v[1] && edgeSet.find(edge) == edgeSet.end())
        {
            edgeSet.insert(edge);
            constraintVec.push_back(edge);
        }
    }
}