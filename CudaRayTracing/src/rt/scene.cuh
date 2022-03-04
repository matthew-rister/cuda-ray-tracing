#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "cuda_error_check.cuh"
#include "cuda_managed.cuh"
#include "rt/intersection.cuh"
#include "rt/material.cuh"
#include "rt/sphere.cuh"

namespace rt {

__global__ void CreateSceneObjects(Intersectable*** objects, int* size) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*objects = new Intersectable*[*size = 5]{
			new Sphere{glm::vec3{ 0.f, -100.5f, -1.f}, 100.f, new Lambertian{glm::vec3{.8f, .8f, 0.f}}},
			new Sphere{glm::vec3{ 0.f,     0.f, -1.f},   .5f, new Lambertian{glm::vec3{.7f, .3f, .3f}}},
			new Sphere{glm::vec3{-1.f,     0.f, -1.f},   .5f, new Dielectric{1.5f}},
			new Sphere{glm::vec3{-1.f,     0.f, -1.f},  -.4f, new Dielectric{1.5f}},
			new Sphere{glm::vec3{ 1.f,     0.f, -1.f},   .5f, new Metal{glm::vec3{.8f, .6f, .2f}}}
		};
	}
}

__global__ void DeleteSceneObjects(Intersectable*** objects) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		delete[] *objects;
	}
}

struct Scene : CudaManaged<Scene> {

	Scene() {
		CreateSceneObjects<<<1, 1>>>(&objects, &size);
		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
	}

	~Scene() {
		DeleteSceneObjects<<<1, 1>>>(&objects);
		CHECK_CUDA_ERRORS_NOTHROW(cudaGetLastError());
		CHECK_CUDA_ERRORS_NOTHROW(cudaDeviceSynchronize());
	}

	Intersectable** objects{};
	int size{};
};

}
