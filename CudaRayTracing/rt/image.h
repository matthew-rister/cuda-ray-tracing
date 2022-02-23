#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include <stb_image_write.h>
#include <fmt/core.h>

namespace rt {

struct Image {

	Image(const int width, const int height, const int channels, const int max_color_value) noexcept
		: width{width},
		  height{height},
		  channels{channels},
		  max_color_value{max_color_value} {}

	void SaveAs(const std::uint8_t* const frame_buffer, const std::string_view filename) const {
		stbi_flip_vertically_on_write(true);
		if (!stbi_write_png(filename.data(), width, height, channels, frame_buffer, width * channels)) {
			throw std::runtime_error{fmt::format("An error occurred while attempting to save {}", filename)};
		}
	}

	int width, height, channels, max_color_value;
};
}
