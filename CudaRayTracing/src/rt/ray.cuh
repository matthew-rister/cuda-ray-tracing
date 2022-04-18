#pragma once

#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

namespace rt {

class Ray {

public:
	__device__ Ray(const glm::vec3& origin, const glm::vec3& direction, const glm::vec3 color = glm::vec3{1.f})
		: origin_{origin}, direction_{glm::normalize(direction)}, color_{color} {}

	__device__ [[nodiscard]] const glm::vec3& origin() const noexcept { return origin_; }
	__device__ [[nodiscard]] const glm::vec3& direction() const noexcept { return direction_; }
	__device__ [[nodiscard]] const glm::vec3& color() const noexcept { return color_; }

	__device__ [[nodiscard]] glm::vec3 PointAt(const float t) const noexcept { return origin_ + t * direction_; }

private:
	glm::vec3 origin_, direction_, color_;
};
}
