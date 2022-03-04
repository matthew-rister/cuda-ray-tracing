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
	Camera** camera,
	Intersectable*** objects,
	int* size,
	float* aspect_ratio,
	int* image_height,
	int* image_width,
	int* samples_per_pixel,
	int* max_depth) {

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*aspect_ratio = 16.f / 9.f;
		*image_height = 400;
		*image_width = static_cast<int>(*aspect_ratio * static_cast<float>(*image_height));
		*camera = new Camera{glm::vec3{3.f, 3.f, 2.f}, glm::vec3{0.f, 0.f, -1.f}, *aspect_ratio, 20.f, 1.f};
		*objects = new Intersectable*[*size = 5]{
			new Sphere{glm::vec3{ 0.f, -100.5f, -1.f}, 100.f, new Lambertian{glm::vec3{.8f, .8f, 0.f}}},
			new Sphere{glm::vec3{ 0.f,     0.f, -1.f},   .5f, new Lambertian{glm::vec3{.1f, .2f, .5f}}},
			new Sphere{glm::vec3{-1.f,     0.f, -1.f},   .5f, new Dielectric{1.5f}},
			new Sphere{glm::vec3{-1.f,     0.f, -1.f},  -.4f, new Dielectric{1.5f}},
			new Sphere{glm::vec3{ 1.f,     0.f, -1.f},   .5f, new Metal{glm::vec3{.8f, .6f, .2f}}}
		};
		*samples_per_pixel = 100;
		*max_depth = 50;
	}
}

__global__ void DeleteSceneObjects(Camera** camera, Intersectable*** objects) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		delete *camera;
		delete[] *objects;
	}
}

struct Scene : CudaManaged<Scene> {

	Scene() {
		CreateSceneObjects<<<1, 1>>>(
			&camera, &objects, &size, &aspect_ratio, &image_height, &image_width, &samples_per_pixel, &max_depth);
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
	int samples_per_pixel, max_depth;
};

}
