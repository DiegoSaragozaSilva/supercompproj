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
    
    std::vector<City> path(numCities);
    std::vector<DistanceData> distanceData = TSP(cities, path, numCities); 

    // Local search
    int seed = 42;
    std::default_random_engine rndEngine(seed);
    std::uniform_int_distribution<int> distPath(0, path.size() - 1);

    std::vector<City> bestPath = path;
    float shortestDistance = getPathDistance(bestPath);

    int performedSearches = 0;
    int totalLocalSearches = numCities * 10;
    while (performedSearches < totalLocalSearches) {
        // Avoid equal indices
        int r1 = distPath(rndEngine), r2 = distPath(rndEngine);
        if (r1 == r2) continue;

        // Order indices
        if (r1 > r2) {
            int tmp = r1;
            r1 = r2;
            r2 = tmp;
        }

        // Swap items
        std::vector<City> testPath = bestPath;
        std::iter_swap(testPath.begin() + r1, testPath.begin() + r2);

        // New path distance
        float pathDistance = getPathDistance(testPath);

        // Get new best distance if possible
        if (pathDistance < shortestDistance) {
            bestPath = testPath;
            shortestDistance = pathDistance;
        }

        // cerr print
        std::cerr << "local: " << shortestDistance << " ";
        for (City city : bestPath)
            std::cerr << city.id << " ";
        std::cerr << '\n';

        performedSearches++;
    }

    std::cout << shortestDistance << " 0" << "\n";
    for (City city : bestPath)
        std::cout << city.id << " ";
    std::cout << std::endl;

    return 1;
}
