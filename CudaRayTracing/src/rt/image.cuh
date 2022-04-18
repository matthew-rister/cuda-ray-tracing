#pragma once

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <stb_image_write.h>
#include <glm/glm.hpp>

#include "cuda_error_check.cuh"
#include "cuda_managed.cuh"

namespace rt {

class Image final : public CudaManaged<Image> {

public:
	using Color = std::uint8_t;
	using Pixel = glm::vec<3, Color>;

	static constexpr int kMaxColorValue = std::numeric_limits<Color>::max();

	Image(const int width, const int height) noexcept : width_{width}, height_{height} {
		CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&frame_buffer_), sizeof(Pixel) * width_* height_));
	}

	~Image() { CHECK_CUDA_ERRORS_NOTHROW(cudaFree(frame_buffer_)); }

	__device__ [[nodiscard]] int width() const noexcept { return width_; }
	__device__ [[nodiscard]] int height() const noexcept { return height_; }

	__device__ Pixel& operator()(const int i, const int j) const noexcept { return frame_buffer_[i * width_ + j]; }

	void SaveAs(const char* const filename) const {
		stbi_flip_vertically_on_write(true);

		if (!stbi_write_png(filename, width_, height_, channels_, frame_buffer_, width_ * channels_)) {
			std::ostringstream oss;
			oss << "An error occurred while attempting to save " << filename;
			throw std::runtime_error{oss.str()};
		}
	}

private:
	int width_, height_, channels_ = Pixel::length();
	Pixel* frame_buffer_ = nullptr;
};
}
