#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "cuda_error_check.cuh"
#include "rt/camera.cuh"
#include "rt/image.cuh"
#include "rt/intersection.cuh"
#include "rt/ray.cuh"
#include "rt/scene.cuh"
#include "rt/sphere.cuh"

using namespace glm;
using namespace rt;
using namespace std;

__device__ vec3 ComputeRayColor(const Ray& ray, const Scene& scene) noexcept {
	Intersection closest_intersection;
	auto hit = false;
	auto t_max = INFINITY;

	for (const auto& object : scene) {
		if (Intersection intersection; object.Intersect(ray, 0.f, t_max, intersection)) {
			closest_intersection = intersection;
			hit = true;
			t_max = intersection.t;
		}
	}

	if (hit) {
		const auto& [point, normal, t, front_facing] = closest_intersection;
		return .5f * normal + vec3{.5f};
	}

	const auto t = .5f * (ray.direction().y + 1.f);
	return (1.f - t) * vec3{1.f} + t * vec3{.5f, .7f, 1.f};
}

__global__ void Render(const Camera& camera, const Image& image, const Scene& scene) {
	const auto i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const auto j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	const auto k = i * image.width() * image.channels() + j * image.channels();

	if (i < image.height() && j < image.width()) {
		const auto u = static_cast<float>(j) / static_cast<float>(image.width() - 1);
		const auto v = static_cast<float>(i) / static_cast<float>(image.height() - 1);
		const auto ray = camera.RayThrough(u, v);
		const auto color = static_cast<float>(image.max_color_value()) * ComputeRayColor(ray, scene);
		image[k] = static_cast<uint8_t>(color.r);
		image[k + 1] = static_cast<uint8_t>(color.g);
		image[k + 2] = static_cast<uint8_t>(color.b);
	}
}

int main() {

	try {
		constexpr auto kAspectRatio = 16.f / 9.f;
		const auto camera = Camera::Make(vec3{0.f}, kAspectRatio);

		constexpr auto kImageHeight = 400;
		constexpr auto kImageWidth = static_cast<int>(kAspectRatio * kImageHeight);
		constexpr auto kColorChannels = 3;
		constexpr auto kMaxColorValue = numeric_limits<uint8_t>::max();
		const auto image = Image::Make(kImageWidth, kImageHeight, kColorChannels, kMaxColorValue);

		const auto scene = Scene::Make(vector{
			Sphere{vec3{0.f, 0.f, -1.f}, .5f},
			Sphere{vec3{0.f, -100.5f, -1.f}, 100.f}
		});

		const dim3 threads{16, 16};
		const dim3 blocks{kImageWidth / threads.x + 1, kImageHeight / threads.y + 1};
		Render<<<blocks, threads>>>(*camera, *image, *scene);

		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

		image->SaveAs("img/ch6.png");
		return EXIT_SUCCESS;

	} catch (exception& e) {
		cerr << e.what();
		cudaDeviceReset();
		return EXIT_FAILURE;
	}
}
