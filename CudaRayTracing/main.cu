#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string_view>

#include <glm/vec3.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "rt/image.h"

#define CHECK_CUDA_ERRORS(error) CheckCudaErrors((error), __FUNCTION__, __FILE__, __LINE__)

void CheckCudaErrors(const cudaError_t error, const std::string_view function, const std::string_view file, const int line) {
	if (error != cudaSuccess) {
		std::ostringstream oss;
		oss << function << " at " << file << ':' << line << " failed with error code " << error;
		throw std::runtime_error{oss.str()};
	}
}

__device__ __host__ constexpr int GetFrameBufferIndex(const int i, const int j, const int width, const int channels) {
	return i * width * channels + j * channels;
}

__global__ void Render(
	std::uint8_t* const frame_buffer,
	const int width,
	const int height,
	const int channels,
	const std::uint8_t max_color_value) {

	const auto i = blockIdx.y * blockDim.y + threadIdx.y;
	const auto j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		const auto index = GetFrameBufferIndex(i, j, width, channels);
		const auto r = max_color_value * static_cast<float>(j) / static_cast<float>(width - 1);
		const auto g = max_color_value * static_cast<float>(i) / static_cast<float>(height - 1);
		const auto b = max_color_value * .2f;
		frame_buffer[index] = static_cast<std::uint8_t>(r);
		frame_buffer[index + 1] = static_cast<std::uint8_t>(g);
		frame_buffer[index + 2] = static_cast<std::uint8_t>(b);
	}
}

int main() {

	try {
		constexpr auto kImageWidth = 256;
		constexpr auto kImageHeight = 256;
		constexpr auto kColorChannels = 3;
		constexpr auto kMaxColorValue = std::numeric_limits<std::uint8_t>::max();
		constexpr auto kPixelCount = kImageWidth * kImageHeight;
		auto image = rt::Image<kColorChannels>{kImageWidth, kImageHeight};

		std::uint8_t* frame_buffer;
		constexpr auto kFrameBufferSize = static_cast<size_t>(kColorChannels * kPixelCount) * sizeof(std::uint8_t);
		CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&frame_buffer), kFrameBufferSize));

		const dim3 threads{16, 16};
		const dim3 blocks{kImageWidth / threads.x + 1, kImageHeight / threads.y + 1};
		Render<<<blocks, threads>>>(frame_buffer, kImageWidth, kImageHeight, kColorChannels, kMaxColorValue);

		CHECK_CUDA_ERRORS(cudaGetLastError());
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

		for (auto i = 0; i < kImageHeight; ++i) {
			for (auto j = 0; j < kImageWidth; ++j) {
				const auto index = GetFrameBufferIndex(i, j, kImageWidth, kColorChannels);
				const auto r = frame_buffer[index];
				const auto g = frame_buffer[index + 1];
				const auto b = frame_buffer[index + 2];
				image(i, j) = {r, g, b};
			}
		}

		image.SaveAs("img/ch2.png");
		CHECK_CUDA_ERRORS(cudaFree(frame_buffer));
		return EXIT_SUCCESS;
	} catch (std::exception& e) {
		std::cerr << e.what();
		cudaDeviceReset();
		return EXIT_FAILURE;
	}
}
