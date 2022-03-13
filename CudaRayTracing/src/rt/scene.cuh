#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "cuda_error_check.cuh"
#include "cuda_managed.cuh"

#include "rt/camera.cuh"
#include "rt/intersection.cuh"
#include "rt/material.cuh"
#include "rt/sphere.cuh"

namespace rt {

__global__ void CreateSceneObjects(
	Camera** camera_ptr,
	Intersectable*** objects_ptr,
	int* size_ptr,
	float* aspect_ratio_ptr,
	int* image_height_ptr,
	int* image_width_ptr,
	int* samples_ptr, 
	int* max_depth_ptr) {

	if (blockIdx.x != 0 || threadIdx.x != 0) return;

	*aspect_ratio_ptr = 16.f / 9.f;
	*image_height_ptr = 1440;
	*image_width_ptr = static_cast<int>(*aspect_ratio_ptr * static_cast<float>(*image_height_ptr));
	*samples_ptr = 500;
	*max_depth_ptr = 50;
	*camera_ptr = new Camera{glm::vec3{13.f, 2.f, 3.f}, glm::vec3{0.f}, *aspect_ratio_ptr, 20.f, .01f, 10.f};

	constexpr auto n = 18;
	auto* objects = *objects_ptr = new Intersectable*[4 + n * n];
	auto& size = *size_ptr = 0;
	objects[size++] = new Sphere{glm::vec3{0.f, -1000.f, 0.f}, 1000.f, new Lambertian{glm::vec3{.5f}}};
	objects[size++] = new Sphere{glm::vec3{0.f, 1.f, 0.f}, 1.f, new Dielectric{1.5f}};
	objects[size++] = new Sphere{glm::vec3{-4.f, 1.f, 0.f}, 1.f, new Lambertian{glm::vec3{.4f, .2f, .1f}}};
	objects[size++] = new Sphere{glm::vec3{4.f, 1.f, 0.f}, 1.f, new Metal{glm::vec3{.7f, .6f, .5f}}};

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
				objects[size++] = new Sphere{center, .2f, material};
			}
		}
	}
}

__global__ void DeleteSceneObjects(Camera** camera, Intersectable*** objects) {
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
	Intersectable** objects;
	int size;
	float aspect_ratio;
	int image_height, image_width;
	int samples, max_depth;
};

}
