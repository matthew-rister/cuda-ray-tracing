#pragma once

#include <glm/glm.hpp>

namespace rt {

class Ray {

public:
	Ray(const glm::dvec3& origin, const glm::dvec3& direction, const glm::dvec3& color = glm::dvec3{1.})
		: origin_{origin}, direction_{glm::normalize(direction)}, color_{color} {}

	[[nodiscard]] const glm::dvec3& origin() const noexcept { return origin_; }
	[[nodiscard]] const glm::dvec3& direction() const noexcept { return direction_; }
	[[nodiscard]] const glm::dvec3& color() const noexcept { return color_; }

	[[nodiscard]] glm::dvec3 PointAt(const double t) const { return origin_ + t * direction_; }

private:
	glm::dvec3 origin_{}, direction_{}, color_{};
};
}
