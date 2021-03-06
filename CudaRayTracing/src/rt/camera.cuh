#pragma once

#include <cuda_runtime_api.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "rt/ray.cuh"

namespace rt {

class Camera {

public:
	__device__ Camera(
		const glm::vec3& look_from, const glm::vec3& look_at, const float aspect_ratio, const float field_of_view_y)
		: origin_{look_from} {

		const auto theta = glm::radians(field_of_view_y);
		const auto viewport_height = 2.f * std::tan(theta / 2.f);
		const auto viewport_width = aspect_ratio * viewport_height;

		const glm::vec3 world_up{0.f, 1.f, 0.f};
		const auto w = glm::normalize(look_from - look_at);
		const auto u = glm::normalize(glm::cross(world_up, w));
		const auto v = glm::normalize(glm::cross(w, u));

		horizontal_ = viewport_width * u;
		vertical_ = viewport_height * v;
		lower_left_corner_ = origin_ - horizontal_ / 2.f - vertical_ / 2.f - w;
	}

	__device__ [[nodiscard]] Ray RayThrough(const float u, const float v) const {
		const auto direction = lower_left_corner_ + u * horizontal_ + v * vertical_ - origin_;
		return Ray{ origin_, direction};
	}

private:
	glm::vec3 origin_;
	glm::vec3 horizontal_, vertical_;
	glm::vec3 lower_left_corner_;
};
}
