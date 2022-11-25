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

struct OutputData {
    std::vector<City> bestPath;
    float bestDistance;
};

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

    std::vector<City> bestPath;
    float bestDistance = INT_MAX;

    int totalSearches = 10 * numCities;
    std::vector<OutputData> outputData(totalSearches);
    #ifdef _OPENMP
        #pragma omp parallel for
        for (int i = 0; i < totalSearches; i++) {
            std::vector<City> shuffledPath = cities;
            std::shuffle(shuffledPath.begin(), shuffledPath.end(), rndEngine); 
            float shuffledDistance = getPathDistance(shuffledPath);
            #pragma omp parallel for
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

            outputData[i].bestPath = bestPath;
            outputData[i].bestDistance = bestDistance;
        }

        for (OutputData data : outputData) {
            std::cerr << "local: " << data.bestDistance << " ";
            for (const auto city : data.bestPath)
                std::cerr << city.id << " ";
            std::cerr << std::endl;
        }
    #else
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

            outputData[i].bestPath = bestPath;
            outputData[i].bestDistance = bestDistance;
        }

        for (OutputData data : outputData) {
            std::cerr << "local: " << data.bestDistance << " ";
            for (const auto city : data.bestPath)
                std::cerr << city.id << " ";
            std::cerr << std::endl;
        }
    #endif

    std::cout << bestDistance << " 0" << "\n";
    for (City city : bestPath)
        std::cout << city.id << " ";
    std::cout << std::endl;

    return 1;
}
