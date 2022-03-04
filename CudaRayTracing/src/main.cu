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

using namespace rt;

namespace {

__device__ glm::vec3 TracePath(Ray ray, const Scene& scene, const int max_depth, curandState_t* random_state) {

	for (auto i = 0; i < max_depth; ++i) {
		Intersection closest_intersection;
		auto t_max = INFINITY;

		for (auto j = 0; j < scene.size; ++j) {
			if (const auto intersection = scene.objects[j]->Intersect(ray, 0.001f, t_max); intersection.hit) {
				closest_intersection = intersection;
				t_max = intersection.t;
			}
		}

		if (closest_intersection.hit) {
			ray = closest_intersection.material->Scatter(ray, closest_intersection, random_state);
		} else {
			const auto t = .5f * (ray.direction().y + 1.f);
			return ray.color() * glm::mix(glm::vec3{1.f}, glm::vec3{.5f, .7f, 1.f}, t);
		}
	}

	return glm::vec3{0.f};
}

__global__ void Render(
	const Image& image, const Camera& camera, const Scene& scene, const int samples_per_pixel, const int max_depth) {
	const auto i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const auto j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	if (i < image.height() && j < image.width()) {
		const auto thread_id = i * image.width() + j;
		curandState_t random_state;
		curand_init(0, thread_id, 0, &random_state);

		glm::vec3 accumulated_color{0.f};
		for (auto k = 0; k < samples_per_pixel; ++k) {
			const auto u = (j + curand_uniform(&random_state)) / (image.width() - 1.f);
			const auto v = (i + curand_uniform(&random_state)) / (image.height() - 1.f);
			const auto ray = camera.RayThrough(u, v);
			accumulated_color += TracePath(ray, scene, max_depth, &random_state);
		}

		const auto average_color = accumulated_color / static_cast<float>(samples_per_pixel);
		const auto gamma_correction = glm::sqrt(average_color);
		image(i, j) = static_cast<float>(Image::kMaxColorValue) * gamma_correction;
	}
}

} // namespace

int main() {

	try {
		constexpr auto kAspectRatio = 16.f / 9.f;
		constexpr auto kFieldOfViewY = 20.f;
		const glm::vec3 look_from{-2.f, 2.f, 1.f};
		const glm::vec3 look_at{0.f, 0.f, -1.f};
		const auto camera = Camera::MakeCudaManaged(look_from, look_at, kAspectRatio, kFieldOfViewY);

		constexpr auto kImageHeight = 400;
		constexpr auto kImageWidth = static_cast<int>(kAspectRatio * kImageHeight);
		const auto image = Image::MakeCudaManaged(kImageWidth, kImageHeight);

		constexpr auto kSamplesPerPixel = 100;
		constexpr auto kMaxDepth = 50;
		const auto scene = Scene::MakeCudaManaged();

		const dim3 threads{16, 16};
		const dim3 blocks{kImageWidth / threads.x + 1, kImageHeight / threads.y + 1};
		Render<<<blocks, threads>>>(*image, *camera, *scene, kSamplesPerPixel, kMaxDepth);
		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

		image->SaveAs("img/ch11.png");

	} catch (std::exception& e) {
		std::cerr << e.what();
		cudaDeviceReset();
		return EXIT_FAILURE;
	}

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
