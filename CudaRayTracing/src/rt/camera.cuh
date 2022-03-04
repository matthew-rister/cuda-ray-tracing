#pragma once

#include <cuda_runtime_api.h>
#include <glm/vec3.hpp>

#include "cuda_managed.cuh"
#include "rt/ray.cuh"

namespace rt {

class Camera final : public CudaManaged<Camera> {

public:
	Camera(const glm::vec3& look_from, const glm::vec3& look_at, const float aspect_ratio, const float field_of_view_y)
		: origin_{look_from} {

		const auto theta = glm::radians(field_of_view_y);
		const auto viewport_height = 2.f * std::tan(theta / 2.f);
		const auto viewport_width = aspect_ratio * viewport_height;

		w_ = glm::normalize(look_from - look_at);
		u_ = glm::normalize(glm::cross(kWorldUp, w_));
		v_ = glm::normalize(glm::cross(w_, u_));

		u_ *= viewport_width;
		v_ *= viewport_height;

		lower_left_corner_ = origin_ - u_ / 2.f - v_ / 2.f - w_;
	}

	__device__ [[nodiscard]] Ray RayThrough(const float u, const float v) const {
		return Ray{origin_, lower_left_corner_ + u * u_ + v * v_ - origin_};
	}

private:
	inline static const glm::vec3 kWorldUp{0.f, 1.f, 0.f};
	glm::vec3 origin_;
	glm::vec3 u_, v_, w_;
	glm::vec3 lower_left_corner_;

};

} // namespace rt
