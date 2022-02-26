#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "cuda_error_check.cuh"
#include "rt/camera.cuh"
#include "rt/image.cuh"
#include "rt/intersection.cuh"
#include "rt/ray.cuh"
#include "rt/scene.cuh"
#include "rt/sphere.cuh"

using namespace rt;

namespace {

__device__ glm::vec3 GetRayColor(const Ray& ray, const Scene& scene) noexcept {
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
		return .5f * normal + glm::vec3{.5f};
	}

	const auto t = .5f * (ray.direction().y + 1.f);
	return (1.f - t) * glm::vec3{1.f} + t * glm::vec3{.5f, .7f, 1.f};
}

__global__ void Render(const Image& image, const Camera& camera, const Scene& scene, const int samples_per_pixel) {
	const auto i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const auto j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

	if (i < image.height() && j < image.width()) {
		const auto thread_id = i * image.width() * image.channels() + j * image.channels();
		curandState_t random_state;
		curand_init(0, thread_id, 0, &random_state);

		glm::vec3 color{0.f};
		for (auto k = 0; k < samples_per_pixel; ++k) {
			const auto du = curand_uniform(&random_state);
			const auto dv = curand_uniform(&random_state);
			const auto u = (static_cast<float>(j) + du) / static_cast<float>(image.width() - 1);
			const auto v = (static_cast<float>(i) + dv) / static_cast<float>(image.height() - 1);
			const auto ray = camera.RayThrough(u, v);
			color += GetRayColor(ray, scene);
		}

		image(i, j) = static_cast<float>(rt::Image::kMaxColorValue) / static_cast<float>(samples_per_pixel) * color;
	}
}

} // namespace

int main() {

	try {
		constexpr auto kAspectRatio = 16.f / 9.f;
		const auto camera = Camera::MakeCudaManaged(glm::vec3{0.f}, kAspectRatio);

		constexpr auto kImageHeight = 400;
		constexpr auto kImageWidth = static_cast<int>(kAspectRatio * kImageHeight);
		const auto image = Image::MakeCudaManaged(kImageWidth, kImageHeight);

		const Sphere sphere1{glm::vec3{0.f, 0.f, -1.f}, .5f};
		const Sphere sphere2{glm::vec3{0.f, -100.5f, -1.f}, 100.f};
		const auto scene = Scene::MakeCudaManaged(std::vector{sphere1, sphere2});

		constexpr auto kSamplesPerPixel = 100;
		const dim3 threads{16, 16};
		const dim3 blocks{kImageWidth / threads.x + 1, kImageHeight / threads.y + 1};
		Render<<<blocks, threads>>>(*image, *camera, *scene, kSamplesPerPixel);

		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

		image->SaveAs("img/ch7.png");
		return EXIT_SUCCESS;

	} catch (std::exception& e) {
		std::cerr << e.what();
		cudaDeviceReset();
		return EXIT_FAILURE;
	}
}
