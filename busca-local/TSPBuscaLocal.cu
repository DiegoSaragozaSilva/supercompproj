#include <cmath>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

struct City {
    int id;
    float x, y;

    __host__ __device__ float operator()(const City& a, const City& b) const {
        return sqrtf((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y));
    }
};

float getPathDistance(thrust::device_vector<City> devicePath) {
    thrust::device_vector<City> devicePathShifted(devicePath.size());
    thrust::copy(devicePath.begin() + 1, devicePath.end(), devicePathShifted.begin());
    devicePathShifted[devicePathShifted.size() - 1] = devicePath[0];

    thrust::device_vector<float> deviceDistances(devicePath.size());
    thrust::transform(devicePath.begin(), devicePath.end(), devicePathShifted.begin(), deviceDistances.begin(), City());
    float sumDistances = thrust::reduce(deviceDistances.begin(), deviceDistances.end(), 0.0f, thrust::plus<float>());
    return sumDistances;    
}

int main() {
    uint32_t numCities = 0;
    std::cin >> numCities;

    thrust::host_vector<City> cities(numCities);
    for (uint32_t i = 0; i < numCities; i++) {
        cities[i].id = i;
        std::cin >> cities[i].x;
        std::cin >> cities[i].y;
    } 

    uint32_t seed = 10;
    thrust::default_random_engine rndEngine(seed);

    thrust::device_vector<City> deviceCities(cities);
    thrust::device_vector<City> deviceBestPath(cities);
    float bestDistance = getPathDistance(deviceBestPath);

    uint32_t totalSearches = numCities * 10;
    for (uint32_t i = 0; i < totalSearches; i++) {
        thrust::device_vector<City> deviceCurrentPath(deviceBestPath);
        thrust::shuffle(deviceCurrentPath.begin(), deviceCurrentPath.end(), rndEngine);
        float currentDistance = getPathDistance(deviceCurrentPath);

        for (uint32_t j = 0; j < numCities - 1; j++) {
            thrust::swap(deviceCurrentPath[j], deviceCurrentPath[j + 1]);
            float newDistance = getPathDistance(deviceCurrentPath);
            if (newDistance < currentDistance) currentDistance = newDistance;
        }

        if (currentDistance < bestDistance) {
            deviceBestPath = deviceCurrentPath;
            bestDistance = currentDistance;
        }
    }

    cities = deviceBestPath;

    std::cout << bestDistance << " 0" << std::endl;
    for (const auto city : cities)
        std::cout << city.id << " ";
    std::cout << std::endl;
}
