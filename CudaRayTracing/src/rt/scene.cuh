#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "cuda_error_check.cuh"
#include "cuda_managed.cuh"

#include "rt/camera.cuh"
#include "rt/hittable.cuh"
#include "rt/material.cuh"
#include "rt/sphere.cuh"

namespace rt {

__global__ void CreateSceneObjects(
	Camera** const camera,
	Hittable*** const objects,
	int* const size,
	float* const aspect_ratio,
	int* const image_height,
	int* const image_width,
	int* const samples,
	int* const max_depth) {

	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	*aspect_ratio = 16.f / 9.f;
	*image_height = 1440;
	*image_width = static_cast<int>(*aspect_ratio * static_cast<float>(*image_height));
	*samples = 500;
	*max_depth = 50;
	*camera = new Camera{glm::vec3{13.f, 2.f, 3.f}, glm::vec3{0.f}, *aspect_ratio, 20.f, .01f, 10.f};

	constexpr auto n = 18;
	auto& k = *size = 0;
	*objects = new Hittable*[4 + n * n];
	(*objects)[k++] = new Sphere{glm::vec3{0.f, -1000.f, 0.f}, 1000.f, new Lambertian{glm::vec3{.5f}}};
	(*objects)[k++] = new Sphere{glm::vec3{0.f, 1.f, 0.f}, 1.f, new Dielectric{1.5f}};
	(*objects)[k++] = new Sphere{glm::vec3{-4.f, 1.f, 0.f}, 1.f, new Lambertian{glm::vec3{.4f, .2f, .1f}}};
	(*objects)[k++] = new Sphere{glm::vec3{4.f, 1.f, 0.f}, 1.f, new Metal{glm::vec3{.7f, .6f, .5f}}};

	curandState_t curand_state;
	curand_init(0, 0, 0, &curand_state);

	for (auto half_n = n / 2, i = -half_n; i < half_n; ++i) {
		for (auto j = -half_n; j < half_n; ++j) {
			const auto cx = i + .9f * curand_uniform(&curand_state);
			const auto cz = j + .9f * curand_uniform(&curand_state);

			if (const glm::vec3 center{cx, .2f, cz}; glm::length(center - glm::vec3{4.f, .2f, 0.f}) > .9f) {
				Material* material;
				if (const auto material_probability = curand_uniform(&curand_state); material_probability < .75f) {
					material = new Lambertian{
						glm::vec3{
							curand_uniform(&curand_state) * curand_uniform(&curand_state),
							curand_uniform(&curand_state) * curand_uniform(&curand_state),
							curand_uniform(&curand_state) * curand_uniform(&curand_state)
						}
					};
				} else if (material_probability < .9f) {
					material = new Metal{
						glm::vec3{.5f} + .5f * glm::vec3{
							curand_uniform(&curand_state),
							curand_uniform(&curand_state),
							curand_uniform(&curand_state)
						},
						.25f * curand_uniform(&curand_state)
					};
				} else {
					material = new Dielectric{1.5f};
				}
				(*objects)[k++] = new Sphere{center, .2f, material};
			}
		}
	}
}

__global__ void DeleteSceneObjects(Camera** camera, Hittable*** objects) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		delete *camera;
		delete[] *objects;
	}
}

struct Scene final : CudaManaged<Scene> {

	Scene() {
		CreateSceneObjects<<<1, 1>>>(
			&camera, &objects, &size, &aspect_ratio, &image_height, &image_width, &samples, &max_depth);
		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	}

	~Scene() {
		DeleteSceneObjects<<<1, 1>>>(&camera, &objects);
		CHECK_CUDA_ERRORS_NOTHROW(cudaGetLastError());
		CHECK_CUDA_ERRORS_NOTHROW(cudaDeviceSynchronize());
	}

	Camera* camera;
	Hittable** objects;
	int size;
	float aspect_ratio;
	int image_height, image_width;
	int samples, max_depth;
};
}
