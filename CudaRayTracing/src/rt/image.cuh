#pragma once

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <cuda_runtime_api.h>
#include <stb_image_write.h>

#include "cuda_error_check.cuh"
#include "cuda_managed.cuh"

namespace rt {

class Image final : public CudaManaged<Image> {

public:
	using color_t = std::uint8_t;
	using pixel_t = glm::vec<3, color_t>;
	static constexpr int kMaxColorValue = std::numeric_limits<color_t>::max();

	__host__ Image(const int width, const int height) noexcept : width_{width}, height_{height} {
		const auto size = sizeof(pixel_t) * width_ * height_;
		CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&frame_buffer_), size));
	}

	__host__ ~Image() { CHECK_CUDA_ERRORS_NOTHROW(cudaFree(frame_buffer_)); }

	__device__ [[nodiscard]] int width() const noexcept { return width_; }
	__device__ [[nodiscard]] int height() const noexcept { return height_; }
	__device__ [[nodiscard]] int channels() const noexcept { return channels_; }

	__device__ pixel_t& operator()(const int i, const int j) const noexcept {
		return frame_buffer_[i * width_ + j];
	}

	__host__ void SaveAs(const std::string_view filename) const {
		stbi_flip_vertically_on_write(true);

		if (!stbi_write_png(filename.data(), width_, height_, channels_, frame_buffer_, width_ * channels_)) {
			std::ostringstream oss;
			oss << "An error occurred while attempting to save " << filename;
			throw std::runtime_error{oss.str()};
		}
	}

private:
	int width_, height_, channels_ = pixel_t::length();
	pixel_t* frame_buffer_{};
};

} // namespace rt
