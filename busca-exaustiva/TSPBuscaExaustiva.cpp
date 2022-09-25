#include <iostream>
#include <cmath>
#include <algorithm>
#include <climits>
#include <vector>
#include <tuple>
#include <random>

struct City {
    int id;
    float x, y;
};

struct DistanceData {
    int dstId;
    float distance;
};

int factorial(int n) {
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

DistanceData distance(City a, City b) {
    float diffX = b.x - a.x;
    float diffY = b.y - a.y;

    DistanceData data;
    data.dstId = b.id;
    data.distance = sqrt(diffX * diffX + diffY * diffY);

    return data;
}

float getPathDistance(std::vector<City> path) {
    float pathDistance = 0;
    for (size_t i = 0; i < path.size() - 1; i++) {
        DistanceData data = distance(path[i], path[i + 1]);
        pathDistance += data.distance;
    }

    pathDistance += distance(path[0], path[path.size() - 1]).distance;
    return pathDistance;
}

std::vector<City> TSPExaustivo(std::vector<City> path, std::vector<City> possibleCities, int* numLeafs) {
    // Leaf node
    if (possibleCities.size() == 0) {
        *numLeafs += 1;
        return path;
    }

    // Exaustive search
    std::vector<std::vector<City>> childrenBestPaths(possibleCities.size());
    for (size_t i = 0; i < possibleCities.size(); i++) {
        std::vector<City> childPath = path;
        childPath.push_back(possibleCities[i]);

        std::vector<City> childPossibleCities = possibleCities;
        childPossibleCities.erase(childPossibleCities.begin() + i);
        childrenBestPaths[i] = TSPExaustivo(childPath, childPossibleCities, numLeafs);
    }

    // Find child best path
    std::vector<City> bestPath;
    float bestDistance = INT_MAX;
    for (const auto &path : childrenBestPaths) {
        float childDistance = getPathDistance(path);
        if (childDistance < bestDistance) {
            bestDistance = childDistance;
            bestPath = path;
        }
    }

    return bestPath;
}

int main() {
    int numCities = 0;
    std::cin >> numCities;

    std::vector<City> cities(numCities);
    for (int i = 0; i < numCities; i++) {
        cities[i].id = i;
        std::cin >> cities[i].x;
        std::cin >> cities[i].y;
    } 

    // Busca exaustiva
    int numLeafs = 0;
    std::vector<City> defaultPath;
    std::vector<City> bestPath = TSPExaustivo(defaultPath, cities, &numLeafs);

    std::cerr << "num_leafs " << numLeafs << std::endl;
    std::cout << getPathDistance(bestPath) << " 1" << std::endl;
    for (const auto &city : bestPath)
        std::cout << city.id << " ";
    std::cout << std::endl;
    //std::vector<int> possibleCities(numCities);
    //for (int i = 0; i < numCities; i++)
    //    possibleCities[i] = i;
    //std::vector<int> defaultPath;
    //std::vector<std::vector<int>> possiblePaths = TSPExaustivo(defaultPath, possibleCities, cities.size());

    //std::vector<City> bestPath;
    //float bestDistance = INT_MAX;
    //for (const auto &path : possiblePaths) {
    //    std::vector<City> possiblePath;
    //    for (const auto &cityId : path)
    //        possiblePath.push_back(cities[cityId]);
    //    float possibleDistance = getPathDistance(possiblePath);
    //    if (possibleDistance < bestDistance) {
    //        bestDistance = possibleDistance;
    //        bestPath = possiblePath;
    //    }
    //}

    //std::cerr << "num_leafs " << possiblePaths.size() << std::endl;
    //std::cout << bestDistance << " 1" << std::endl;
    //for (const auto& city : bestPath)
    //    std::cout << city.id << " ";
    //std::cout << std::endl;

    return 1;
}
