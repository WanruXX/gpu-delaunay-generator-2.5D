#include "../include/IOType.h"
#include "../include/HashFunctors.h"
#include <unordered_set>

namespace gdg
{
void Input::removeDuplicates()
{
    Point2DHVec oldPointVec      = std::move(pointVec);
    EdgeHVec    oldConstraintVec = std::move(constraintVec);

    std::unordered_set<Point, PointHash> pointSet;
    std::vector<int>                     pointMap(oldPointVec.size());
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
        pointMap[i] = static_cast<int>(pointVec.size()) - 1;
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

void Output::reset()
{
    triVec.clear();
    triOppVec.clear();
    edgeSet.clear();
    infPt = {0, 0, 0};
}

void Output::getEdgesFromTriVec()
{
    for (auto &tri : triVec)
    {
        edgeSet.insert({tri._v[0], tri._v[1]});
        edgeSet.insert({tri._v[0], tri._v[2]});
        edgeSet.insert({tri._v[1], tri._v[2]});
    }
}
}