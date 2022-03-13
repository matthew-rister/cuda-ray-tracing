#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <ranges>

#include <glm/glm.hpp>

#include "rt/image.h"

namespace rt {

class Texture {

public:
	[[nodiscard]] virtual glm::dvec3 ColorAt(double u, double v, const glm::dvec3& point) const = 0;

protected:
	~Texture() = default;
};

class Texturable {
public:
	[[nodiscard]] virtual glm::dvec2 TextureCoordinatesAt(const glm::dvec3& point) const = 0;

protected:
	~Texturable() = default;
};

class Checkerboard final : public Texture {

public:
	Checkerboard(const double width, const double height) noexcept
		: Checkerboard{width, height, glm::dvec3{0.}, glm::dvec3{1.}} {}

	Checkerboard(
		const double width, const double height, const glm::dvec3& even_color, const glm::dvec3& odd_color) noexcept
		: width_{width}, height_{height}, even_color_{even_color}, odd_color_{odd_color} {}

	[[nodiscard]] glm::dvec3 ColorAt(const double u, const double v, const glm::dvec3&) const noexcept override {
		const auto s = static_cast<int>(width_ * u);
		const auto t = static_cast<int>(height_ * v);
		return ((s + t) % 2) ? even_color_ : odd_color_;
	}

private:
	double width_, height_;
	glm::dvec3 even_color_{0.}, odd_color_{1.};
};

template <int Channels = 3>
class Image2d final : public Texture {

public:
	explicit Image2d(const std::string_view filename) : image_{filename} {}

	[[nodiscard]] glm::dvec3 ColorAt(const double u, const double v, const glm::dvec3&) const override {
		const auto i = static_cast<int>(std::clamp(v, 0., 1.) * (image_.height() - 1.));
		const auto j = static_cast<int>(std::clamp(u, 0., 1.) * (image_.width() - 1.));
		return 1. / image_.kMaxColorValue * glm::dvec3{image_(i, j)};
	}

private:
	Image<Channels> image_;
};

class PerlinNoise final : public Texture {
	static constexpr int kSampleSize = 256;
	static constexpr int kBitmask = kSampleSize - 1;

public:
	explicit PerlinNoise(const double frequency = 1.)
		: x_{MakeRandomPermutation()}, y_{MakeRandomPermutation()}, z_{MakeRandomPermutation()}, frequency_{frequency} {
		std::ranges::generate(noise_, [&] { return glm::sphericalRand(1.); });
	}

	[[nodiscard]] glm::dvec3 ColorAt(const double, const double, const glm::dvec3& point) const override {
		return glm::dvec3{.5 * (1. + std::sin(frequency_ * point.z + 10. * TurbulenceAt(point)))};
	}

private:
	static std::array<int, kSampleSize> MakeRandomPermutation() {
		static std::default_random_engine random_engine;
		std::array<int, kSampleSize> permutation{};
		std::iota(permutation.begin(), permutation.end(), 0);
		std::ranges::shuffle(permutation, random_engine);
		return permutation;
	}

	[[nodiscard]] double TurbulenceAt(glm::dvec3 point, const int depth = 7) const {
		auto uniform_color = 0.;
		auto weight = 1.;

		for (auto i = 0; i < depth; ++i) {
			uniform_color += weight * NoiseAt(point);
			weight *= .5;
			point *= 2;
		}

		return std::abs(uniform_color);
	}

	[[nodiscard]] double NoiseAt(const glm::dvec3& point) const {

		const auto x_floor = static_cast<int>(std::floor(point.x));
		const auto y_floor = static_cast<int>(std::floor(point.y));
		const auto z_floor = static_cast<int>(std::floor(point.z));

		auto u = point.x - x_floor;
		auto v = point.y - y_floor;
		auto w = point.z - z_floor;

		// hermite smoothing
		u = u * u * (3. - 2. * u);
		v = v * v * (3. - 2. * v);
		w = w * w * (3. - 2. * w);

		// trilinear interpolation
		auto uniform_color = 0.;
		for (auto i = 0; i < 2; ++i) {
			for (auto j = 0; j < 2; ++j) {
				for (auto k = 0; k < 2; ++k) {
					const glm::dvec3 weight{u - i, v - j, w - k};
					const auto u_interpolation = glm::mix(1. - u, u, static_cast<double>(i));
					const auto v_interpolation = glm::mix(1. - v, v, static_cast<double>(j));
					const auto w_interpolation = glm::mix(1. - w, w, static_cast<double>(k));
					const auto index = (i + x_floor & kBitmask) ^ (j + y_floor & kBitmask) ^ (k + z_floor & kBitmask);
					const auto weighted_noise = glm::dot(weight, noise_.at(index));
					uniform_color += u_interpolation * v_interpolation * w_interpolation * weighted_noise;
				}
			}
		}

		return uniform_color;
	}

	std::array<glm::dvec3, kSampleSize> noise_{};
	std::array<int, kSampleSize> x_, y_, z_;
	double frequency_;
};
}
