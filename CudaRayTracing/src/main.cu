#include <chrono>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "cuda_error_check.cuh"
#include "cuda_managed.cuh"
#include "rt/camera.cuh"
#include "rt/image.cuh"
#include "rt/intersection.cuh"
#include "rt/material.cuh"
#include "rt/ray.cuh"
#include "rt/scene.cuh"

using namespace glm;
using namespace rt;
using namespace std;

namespace {

__device__ vec3 TracePath(const Scene& scene, Ray ray, curandState_t* random_state) {

	for (auto i = 0; i < scene.max_depth; ++i) {
		Intersection closest_intersection;
		Hittable* closest_object = nullptr;
		auto t_max = INFINITY;

		for (auto j = 0; j < scene.size; ++j) {
			if (const auto intersection = scene.objects[j]->Intersect(ray, 0.001f, t_max); intersection.hit) {
				closest_intersection = intersection;
				closest_object = scene.objects[j];
				t_max = intersection.t;
			}
		}

		if (closest_object) {
			ray = closest_object->Material()->Scatter(ray, closest_intersection, random_state);
		} else {
			const auto t = .5f * (ray.Direction().y + 1.f);
			return ray.Color() * mix(vec3{1.f}, vec3{.5f, .7f, 1.f}, t);
		}
	}

	return vec3{0.f};
}

__global__ void Render(const Scene& scene, const Image& image) {
	const auto i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const auto j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	if (i < image.Height() && j < image.Width()) {
		const auto thread_id = i * image.Width() + j;
		curandState_t random_state;
		curand_init(0, thread_id, 0, &random_state);

		vec3 accumulated_color{0.f};
		for (auto k = 0; k < scene.samples; ++k) {
			const auto u = (j + curand_uniform(&random_state)) / (image.Width() - 1.f);
			const auto v = (i + curand_uniform(&random_state)) / (image.Height() - 1.f);
			const auto ray = scene.camera->RayThrough(u, v);
			accumulated_color += TracePath(scene, ray, &random_state);
		}

		const auto average_color = accumulated_color / static_cast<float>(scene.samples);
		const auto gamma_correction = sqrt(average_color);
		image(i, j) = static_cast<float>(Image::kMaxColorValue) * gamma_correction;
	}
}
}

int main() {

	try {
		const auto start_time = chrono::high_resolution_clock::now();
		const auto scene = Scene::MakeCudaManaged();
		const auto image = Image::MakeCudaManaged(scene->image_width, scene->image_height);

		const dim3 threads{16, 16};
		const dim3 blocks{scene->image_width / threads.x + 1, scene->image_height / threads.y + 1};
		Render<<<blocks, threads>>>(*scene, *image);
		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

		image->SaveAs("img/output.png");
		const auto end_time = chrono::high_resolution_clock::now();
		cout << "Image rendered in " << chrono::duration<double>{end_time - start_time}.count() << " seconds\n";

	} catch (exception& e) {
		cerr << e.what() << endl;
		cudaDeviceReset();
		return EXIT_FAILURE;
	}

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
