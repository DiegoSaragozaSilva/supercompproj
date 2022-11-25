#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <climits>

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

DistanceData getMinDistanceData(DistanceData* data, int dataSize) {
    DistanceData minData;
    minData.dstId = 0;
    minData.distance = INT_MAX;
    for (int i = 0; i < dataSize; i++) {
        if (minData.distance > data[i].distance) {
            minData.dstId = data[i].dstId;
            minData.distance = data[i].distance; 
        }
    }
    return minData;
}

float TSP(City* cities, int* path, int numCities) {
    float totalDistance = 0;
    int pathSize = 0;
    path[pathSize++] = 0;

    while (pathSize < numCities) {
        int i = 0, j = 0;
        DistanceData* distances = (DistanceData*)malloc(sizeof(DistanceData) * (numCities - pathSize));
        #ifdef _OPENMP
            #pragma omp parallel for
            for (i = 0; i < numCities; i++) {
                if (std::find(path, (int*)(path + pathSize), i) == (int*)(path + pathSize))
                    distances[j++] = distance(cities[path[pathSize - 1]], cities[i]);
            }
        #else
            for (i = 0; i < numCities; i++) {
                if (std::find(path, (int*)(path + pathSize), i) == (int*)(path + pathSize))
                    distances[j++] = distance(cities[path[pathSize - 1]], cities[i]);
            }       
        #endif
        
        DistanceData minDistanceData = getMinDistanceData(distances, j);
        totalDistance += minDistanceData.distance;
        path[pathSize++] = minDistanceData.dstId;
        free(distances);
    }

    DistanceData backDistance = distance(cities[path[pathSize - 1]], cities[0]);
    return totalDistance + backDistance.distance;
}

int main() {
    int numCities = 0;
    std::cin >> numCities;

    City* cities = (City*)malloc(sizeof(City) * numCities); 
    for (int i = 0; i < numCities; i++) {
        cities[i].id = i;
        std::cin >> cities[i].x;
        std::cin >> cities[i].y;
    }
    
    int* path = (int*)malloc(sizeof(int) * numCities);
    float totalDistance = TSP(cities, path, numCities); 

    std::cout << totalDistance << " 0" << "\n";
    for (int i = 0; i < numCities; i++)
        std::cout << path[i] << " ";
    std::cout << "\n";
    
    free(cities);
    free(path);

    return 1;
}
