#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <glm/vec3.hpp>

#pragma warning(disable:4996)
#include "stb_image_write.h"
#pragma warning(default:4996)

namespace rt {

template <int Channels>
class Image {

public:
	Image(const int width, const int height) noexcept
		: width_{width}, height_{height}, pixels_{decltype(pixels_)(static_cast<std::size_t>(width) * height)} {}

	[[nodiscard]] int width() const noexcept { return width_; }
	[[nodiscard]] int height() const noexcept { return height_; }

	glm::vec<Channels, std::uint8_t>& operator()(const int i, const int j) { return pixels_.at(i * width_ + j); }

	void SaveAs(const std::string_view filename) const {
		stbi_flip_vertically_on_write(true);
		if (!stbi_write_png(filename.data(), width_, height_, Channels, pixels_.data(), width_ * Channels)) {
			std::ostringstream oss;
			oss << "Failed to save " << filename;
			throw std::runtime_error{oss.str()};
		}
	}

private:
	int width_, height_;
	std::vector<glm::vec<Channels, std::uint8_t>> pixels_;
};
}
