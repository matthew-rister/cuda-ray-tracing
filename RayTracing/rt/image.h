#pragma once

#include <cstdint>

#include <format>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#pragma warning(disable:4996)
#include "stb_image.h"
#include "stb_image_write.h"
#pragma warning(default:4996)

namespace rt {

template <int Channels = 3>
class Image {

public:
	static constexpr std::uint8_t kMaxColorValue = std::numeric_limits<std::uint8_t>::max();

	Image(const int width, const int height) noexcept
		: width_{width}, height_{height}, pixels_{decltype(pixels_)(static_cast<std::size_t>(width) * height)} {}

	explicit Image(const std::string_view filename) {
		stbi_set_flip_vertically_on_load(true);
		int channels;
		auto* const data = stbi_load(filename.data(), &width_, &height_, &channels, Channels);

		if (!data) throw std::runtime_error{std::format("Failed to load {}", filename)};
		if (channels != Channels) throw std::runtime_error{"Unsupported image format"};

		const auto size = static_cast<std::size_t>(width_) * height_;
		pixels_.reserve(size);

		for (auto i = 0, k = 0; i < size; ++i) {
			glm::vec<Channels, std::uint8_t> pixel{};
			for (auto j = 0; j < Channels; ++j, ++k) {
				pixel[j] = *(data + k);
			}
			pixels_.push_back(std::move(pixel));
		}

		stbi_image_free(data);
	}

	[[nodiscard]] int width() const noexcept { return width_; }
	[[nodiscard]] int height() const noexcept { return height_; }

	glm::vec<Channels, std::uint8_t>& operator()(const int i, const int j) {
		return pixels_.at(i * width_ + j);
	}

	[[nodiscard]] const auto& operator()(const int i, const int j) const {
		return const_cast<Image&>(*this)(i, j);
	}

	void SaveAs(const std::string_view filename) const {
		stbi_flip_vertically_on_write(true);
		if (!stbi_write_png(filename.data(), width_, height_, Channels, pixels_.data(), width_ * Channels)) {
			throw std::runtime_error{std::format("Failed to save {}", filename)};
		}
	}

private:
	int width_{}, height_{};
	std::vector<glm::vec<Channels, std::uint8_t>> pixels_;
};
}
