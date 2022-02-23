#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string_view>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fmt/core.h>
#include <glm/glm.hpp>

#include "rt/camera.h"
#include "rt/image.h"
#include "rt/ray.h"
#include "rt/sphere.h"

using namespace fmt;
using namespace glm;
using namespace rt;
using namespace std;

#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#define CHECK_CUDA_ERRORS(status) CheckCudaErrors((status), #status, __FILENAME__, __LINE__)

void CheckCudaErrors(
	const cudaError_t status, const std::string_view function, const std::string_view filename, const int line_number) {

	if (status != cudaSuccess) {
		throw runtime_error{
			format("{} failed at {}:{} with error \"{}\"", function, filename, line_number, cudaGetErrorString(status))
		};
	}
}

__device__ vec3 ComputeRayColor(const Ray& ray, const Sphere& sphere) noexcept {
	if (sphere.Intersect(ray)) return vec3{1.f, 0.f, 0.f};
	const auto t = .5f * (ray.direction().y + 1.f);
	return (1.f - t) * vec3{1.f} + t * vec3{.5f, .7f, 1.f};
}

__global__ void Render(const Image image, const Camera camera, const Sphere sphere, uint8_t* const frame_buffer) {
	const auto [width, height, channels, max_color_value] = image;
	const auto i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const auto j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	const auto k = i * width * channels + j * channels;

	if (i < height && j < width) {
		const auto u = static_cast<float>(j) / static_cast<float>(width - 1);
		const auto v = static_cast<float>(i) / static_cast<float>(height - 1);
		const auto ray = camera.RayThrough(u, v);
		const auto color = static_cast<float>(max_color_value) * ComputeRayColor(ray, sphere);
		frame_buffer[k] = static_cast<uint8_t>(color.r);
		frame_buffer[k + 1] = static_cast<uint8_t>(color.g);
		frame_buffer[k + 2] = static_cast<uint8_t>( color.b);
	}
}

int main() {

	try {
		constexpr auto kAspectRatio = 16.f / 9.f;
		const Camera camera{vec3{0.f}, kAspectRatio};

		constexpr auto kImageHeight = 400;
		constexpr auto kImageWidth = static_cast<int>(kAspectRatio * kImageHeight);
		constexpr auto kColorChannels = 3;
		constexpr auto kImageSize = static_cast<int64_t>(kImageWidth) * kImageHeight * kColorChannels;
		constexpr auto kImageSizeBytes = kImageSize * sizeof(uint8_t);
		constexpr auto kMaxColorValue = numeric_limits<uint8_t>::max();
		const Image image{kImageWidth, kImageHeight, kColorChannels, kMaxColorValue};
		uint8_t* device_frame_buffer = nullptr;
		CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&device_frame_buffer), kImageSizeBytes));

		const glm::vec3 origin{0.f, 0.f, -1.f};
		const Sphere sphere{origin, .5f};

		const dim3 threads{16, 16};
		const dim3 blocks{kImageWidth / threads.x + 1, kImageHeight / threads.y + 1};
		Render<<<blocks, threads>>>(image, camera, sphere, device_frame_buffer);

		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

		const unique_ptr<uint8_t[]> host_frame_buffer{new uint8_t[kImageSize]};
		CHECK_CUDA_ERRORS(cudaMemcpy(host_frame_buffer.get(), device_frame_buffer, kImageSizeBytes, cudaMemcpyDefault));
		CHECK_CUDA_ERRORS(cudaFree(device_frame_buffer));

		image.SaveAs(host_frame_buffer.get(), "img/ch5.png");
		return EXIT_SUCCESS;

	} catch (exception& e) {
		cerr << e.what();
		cudaDeviceReset();
		return EXIT_FAILURE;
	}
}
