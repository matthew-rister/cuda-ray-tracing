#pragma once

#include <cuda_runtime_api.h>
#include <glm/vec3.hpp>

#include "cuda_managed.cuh"
#include "rt/ray.cuh"

namespace rt {

class Camera final : public CudaManaged<Camera> {

public:
	Camera(const glm::vec3& origin, const float aspect_ratio) noexcept
		: origin_{origin},
		  viewport_width_{aspect_ratio * kViewportHeight},
		  lower_left_corner_{origin - glm::vec3{viewport_width_ / 2.f, kViewportHeight / 2.f, kFocalLength}} {}

	__device__ [[nodiscard]] Ray RayThrough(const float u, const float v) const {
		return Ray{origin_, lower_left_corner_ + glm::vec3{u * viewport_width_, v * kViewportHeight, 0.f}};
	}

private:
	static constexpr float kViewportHeight = 2.f;
	static constexpr float kFocalLength = 1.f;
	glm::vec3 origin_;
	float viewport_width_;
	glm::vec3 lower_left_corner_;
};

} // namespace rt
