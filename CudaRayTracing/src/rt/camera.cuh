#pragma once

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "rt/ray.cuh"

namespace rt {

class Camera {

public:
	__device__ Camera(
		const glm::vec3& look_from,
		const glm::vec3& look_at,
		const float aspect_ratio,
		const float field_of_view_y,
		const float aperture = 0.f,
		const float focus_distance = 1.f)
		: origin_{look_from},
		  radius_{aperture / 2.f} {

		const auto theta = glm::radians(field_of_view_y);
		const auto viewport_height = 2.f * std::tan(theta / 2.f);
		const auto viewport_width = aspect_ratio * viewport_height;

		const glm::vec3 world_up{0.f, 1.f, 0.f};
		const auto w = glm::normalize(look_from - look_at);
		const auto u = glm::normalize(glm::cross(world_up, w));
		const auto v = glm::normalize(glm::cross(w, u));

		horizontal_ = focus_distance * viewport_width * u;
		vertical_ = focus_distance * viewport_height * v;
		lower_left_corner_ = origin_ - horizontal_ / 2.f - vertical_ / 2.f - focus_distance * w;
	}

	__device__ [[nodiscard]] Ray RayThrough(const float u, const float v, curandState_t* random_state) const {
		const auto offset = radius_ * MakeRandomVectorInUnitDisk(random_state);
		const auto origin = origin_ + offset.x * horizontal_ + offset.y * vertical_;
		const auto direction = lower_left_corner_ + u * horizontal_ + v * vertical_ - origin;
		return Ray{origin, direction};
	}

private:
	__device__ [[nodiscard]] static glm::vec2 MakeRandomVectorInUnitDisk(curandState_t* random_state) {
		glm::vec2 v;
		do {
			const auto x = curand_uniform(random_state);
			const auto y = curand_uniform(random_state);
			v = 2.f * glm::vec2{x, y} - 1.f;
		} while (glm::length2(v) > 1.f);
		return v;
	}

	glm::vec3 origin_;
	glm::vec3 horizontal_, vertical_;
	glm::vec3 lower_left_corner_;
	float radius_;
};

}
