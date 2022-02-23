#pragma once

#include <cuda_runtime.h>
#include <glm/vec3.hpp>

#include "rt/ray.h"

namespace rt {

class Camera {

public:
	Camera(const glm::vec3& origin, const float aspect_ratio)
		: origin_{origin},
		  horizontal_{aspect_ratio * kViewportHeight, 0.f, 0.f},
		  vertical_{0.f, kViewportHeight, 0.f},
		  focal_{0.f, 0.f, kFocalLength},
		  lower_left_corner_{origin - horizontal_ / 2.f - vertical_ / 2.f - focal_} {}

	__device__ [[nodiscard]] Ray RayThrough(const float u, const float v) const {
		return Ray{origin_, lower_left_corner_ + u * horizontal_ + v * vertical_ - origin_};
	}

private:
	static constexpr float kViewportHeight = 2.f;
	static constexpr float kFocalLength = 1.f;
	glm::vec3 origin_, horizontal_, vertical_, focal_, lower_left_corner_;
};
}