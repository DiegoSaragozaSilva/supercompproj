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

std::vector<std::vector<int>> TSPExaustivo(std::vector<int> path, std::vector<int> possibleCities, int numCities) {
    // Leaf node
    std::vector<std::vector<int>> possiblePaths;
    if (path.size() == (size_t)numCities) {
        possiblePaths.push_back(path);
        return possiblePaths;
    }

    possiblePaths.resize(factorial(possibleCities.size()));

    // Exaustive search
    int i = 0, j = 0, k = factorial(possibleCities.size() - 1), f = k;
    for (const auto &city : possibleCities) {
        std::vector<int> addedPath = path;
        std::vector<int> newPossibleCities = possibleCities;
        newPossibleCities.erase(newPossibleCities.begin() + i);
        addedPath.push_back(city);
        std::vector<std::vector<int>> newPaths = TSPExaustivo(addedPath, newPossibleCities, numCities);
        
        for (j; j < k; j++)
            possiblePaths[j] = newPaths[j % f];
        k += f;
        i++;
    }
    return possiblePaths;
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
    std::vector<int> possibleCities(numCities);
    for (int i = 0; i < numCities; i++)
        possibleCities[i] = i;
    std::vector<int> defaultPath;
    std::vector<std::vector<int>> possiblePaths = TSPExaustivo(defaultPath, possibleCities, cities.size());

    std::vector<City> bestPath;
    float bestDistance = INT_MAX;
    for (const auto &path : possiblePaths) {
        std::vector<City> possiblePath;
        for (const auto &cityId : path)
            possiblePath.push_back(cities[cityId]);
        float possibleDistance = getPathDistance(possiblePath);
        if (possibleDistance < bestDistance) {
            bestDistance = possibleDistance;
            bestPath = possiblePath;
        }
    }

    std::cerr << "num_leafs " << possiblePaths.size() << std::endl;
    std::cout << bestDistance << " 1" << std::endl;
    for (const auto& city : bestPath)
        std::cout << city.id << " ";
    std::cout << std::endl;

    return 1;
}
