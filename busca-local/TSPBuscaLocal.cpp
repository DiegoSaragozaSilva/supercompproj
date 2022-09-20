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

DistanceData distance(City a, City b) {
    float diffX = b.x - a.x;
    float diffY = b.y - a.y;

    DistanceData data;
    data.dstId = b.id;
    data.distance = sqrt(diffX * diffX + diffY * diffY);

    return data;
}

std::tuple<DistanceData, size_t> getMinDistanceData(std::vector<DistanceData> data) {
    DistanceData minData;
    minData.dstId = 0;
    minData.distance = INT_MAX;
    size_t minIndex = 0;

    for (size_t i = 0; i < data.size(); i++) {
        if (minData.distance > data[i].distance) {
            minData.dstId = data[i].dstId;
            minData.distance = data[i].distance; 
            minIndex = i;
        }
    }
    return std::tuple<DistanceData, size_t>(minData, minIndex);
}

std::vector<DistanceData> TSP(std::vector<City> cities, std::vector<City>& path, int numCities) {
    int pathSize = 0;
    path[pathSize++] = cities[0];
    cities.erase(cities.begin());

    std::vector<DistanceData> distanceData(numCities + 1);
    while (pathSize < numCities) {
        std::vector<DistanceData> distances(cities.size());
        for (size_t i = 0; i < cities.size(); i++)
            distances[i] = distance(path[pathSize - 1], cities[i]);

        std::tuple<DistanceData, size_t> minDistanceData = getMinDistanceData(distances);
        DistanceData minData = std::get<0>(minDistanceData);
        size_t dataIndex = std::get<1>(minDistanceData);

        distanceData[pathSize - 1] = minData;
        path[pathSize++] = cities[dataIndex];
        cities.erase(cities.begin() + dataIndex);
    }
    distanceData[distanceData.size() - 1] = distance(path[0], path[path.size() - 1]);
    return distanceData;
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

int main() {
    int numCities = 0;
    std::cin >> numCities;

    std::vector<City> cities(numCities);
    for (int i = 0; i < numCities; i++) {
        cities[i].id = i;
        std::cin >> cities[i].x;
        std::cin >> cities[i].y;
    } 

    // Local search
    int seed = 10;
    std::default_random_engine rndEngine(seed);
    std::uniform_int_distribution<int> distPath(0, cities.size() - 1);

    std::vector<City> bestPath;
    float bestDistance = INT_MAX;

    int totalSearches = 10 * numCities;
    for (int i = 0; i < totalSearches; i++) {
        std::vector<City> shuffledPath = cities;
        std::shuffle(shuffledPath.begin(), shuffledPath.end(), rndEngine); 
        float shuffledDistance = getPathDistance(shuffledPath);
        for (size_t j = 0; j < shuffledPath.size(); j++) {
            for (size_t w = 0; w < shuffledPath.size(); w++) {
                std::vector<City> changePath = shuffledPath;
                std::iter_swap(changePath.begin() + j, changePath.begin() + w);
                float changeDistance = getPathDistance(changePath);
                if (changeDistance < shuffledDistance) {
                    shuffledDistance = changeDistance;
                    shuffledPath = changePath;
                }
            }
        }
        if (shuffledDistance < bestDistance) {
            bestDistance = shuffledDistance;
            bestPath = shuffledPath;
        }

        std::cerr << "local: " << bestDistance << " ";
        for (const auto city : bestPath)
            std::cerr << city.id << " ";
        std::cerr << std::endl;
    }

    std::cout << bestDistance << " 0" << "\n";
    for (City city : bestPath)
        std::cout << city.id << " ";
    std::cout << std::endl;

    return 1;
}
