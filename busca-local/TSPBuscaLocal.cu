#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <random>

struct City {
	int index;
	double x;
	double y;

	__host__ __device__ double operator()(const City &c1, const City &c2) const {
		double dx = c1.x - c2.x;
		double dy = c1.y - c2.y;
		return sqrt(dx * dx + dy * dy);
	}
};

__device__ double get_path_distance(City c1, City c2) {
    double dx = c1.x - c2.x;
    double dy = c1.y - c2.y;
    return sqrt(dx * dx + dy * dy);
}

__global__ void tsp(City* path, int n, double* distance) {
    for (int j = 0; j < n - 1; j++) {
        City aux = path[j];
        path[j] = path[j + 1];
        path[j + 1] = aux;

        City* shifted_path = (City*)malloc(n * sizeof(City));
        for (int i = 0; i < n; i++) shifted_path[i] = path[i];
        shifted_path[n - 1] = path[0];

        double path_distance;
        for (int i = 0; i < n - 1; i++)
            path_distance += get_path_distance(shifted_path[i], shifted_path[i + 1]);
        path_distance += get_path_distance(shifted_path[0], shifted_path[n - 1]);

        if (path_distance < *distance) *distance = path_distance;
        else {
            City aux = path[j];
            path[j] = path[j + 1];
            path[j + 1] = aux;
        }
        
        free(shifted_path);
    }
}

int main() {
	int N;
	std::cin >> N;
	std::vector<City> cities (N);
	for (int i = 0; i < N; ++i) {
		double x, y;
		std::cin >> x;
		std::cin >> y;

		City c = {i, x, y};
		cities[i] = c;
	}

	int seed = 42;
	std::default_random_engine rndEngine(seed);

	// std::cout << "A" << std::endl;
	std::vector<std::vector<City>> tours (10 * N);
	for (int i = 0; i < 10 * N; ++i) {
		std::shuffle(cities.begin(), cities.end(), rndEngine);
		tours[i] = cities;
    }

	// std::cout << "B" << std::endl;
    std::vector<City*> gpuPaths (10 * N);
    for (int i = 0; i < 10 * N; i++) {
        cudaMalloc((void**)&gpuPaths[i], sizeof(City) * N);
        cudaMemcpy(gpuPaths[i], tours[i].data(), sizeof(City) * N, cudaMemcpyHostToDevice);
    } 
	// std::cout << "C" << std::endl;

    std::vector<double*> gpuDistances (10 * N);
    std::vector<cudaStream_t> streams (10 * N);
    for (int i = 0; i < 10 * N; i++) {
        double maxDouble = std::numeric_limits<double>::max();
        cudaMalloc((void**)&gpuDistances[i], sizeof(double));
        cudaMemcpy(gpuDistances[i], &maxDouble, sizeof(double), cudaMemcpyHostToDevice);

        cudaStreamCreate(&streams[i]);
        tsp <<<1, 1, 0, streams[i]>>> (gpuPaths[i], N, gpuDistances[i]);
    }
    cudaDeviceSynchronize();

	// std::cout << "D" << std::endl;
    for (int i = 0; i < 10 * N; i++) cudaFree(gpuPaths[i]);

	// std::cout << "E" << std::endl;
    double best_distance = std::numeric_limits<double>::max();
    for (int i = 0; i < 10 * N; i++) {
        double current_distance = 0.0;
        cudaMemcpy(&current_distance, gpuDistances[i], sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(gpuDistances[i]);
        if (current_distance < best_distance) best_distance = current_distance;
    }

	// std::cout << "F" << std::endl;
    // std::cout << best_distance << std::endl;
}
