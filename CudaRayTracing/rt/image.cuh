#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <cuda_runtime.h>
#include <stb_image_write.h>

#include "cuda_error_check.cuh"
#include "cuda_managed.cuh"

namespace rt {

class Image final : public CudaManaged<Image> {

public:
	__host__ Image(const int width, const int height, const int channels, const int max_color_value) noexcept
		: width_{width}, height_{height}, channels_{channels}, max_color_value_{max_color_value} {
		const auto size = sizeof(uint8_t) * width_ * height_ * channels_;
		CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&frame_buffer_), size));
	}

	__host__ ~Image() { CHECK_CUDA_ERRORS_NOTHROW(cudaFree(frame_buffer_)); }

	__device__ [[nodiscard]] int width() const noexcept { return width_; }
	__device__ [[nodiscard]] int height() const noexcept { return height_; }
	__device__ [[nodiscard]] int channels() const noexcept { return channels_; }
	__device__ [[nodiscard]] int max_color_value() const noexcept { return max_color_value_; }

	__device__ uint8_t& operator[](const int i) const noexcept { return frame_buffer_[i]; }

	__host__ void SaveAs(const std::string_view filename) const {
		stbi_flip_vertically_on_write(true);

		if (!stbi_write_png(filename.data(), width_, height_, channels_, frame_buffer_, width_ * channels_)) {
			std::ostringstream oss;
			oss << "An error occurred while attempting to save " << filename;
			throw std::runtime_error{oss.str()};
		}
	}

private:
	int width_, height_, channels_, max_color_value_;
	std::uint8_t* frame_buffer_{};
};

} // namespace rt
