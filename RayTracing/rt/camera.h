#pragma once

#include <glm/glm.hpp>

#include "rt/ray.h"

namespace rt {

class Camera {

public:
	Camera(const glm::dvec3& look_from, const glm::dvec3& look_at, const double aspect_ratio, const double field_of_view_y)
		: origin_{look_from} {

		const auto theta = glm::radians(field_of_view_y);
		const auto viewport_height = 2. * glm::tan(theta / 2.);
		const auto viewport_width = viewport_height * aspect_ratio;

		w_ = normalize(look_from - look_at);
		u_ = glm::cross(kWorldUp, w_);
		v_ = glm::cross(w_, u_);

		u_ *= viewport_width;
		v_ *= viewport_height;

		lower_left_corner_ = origin_ - u_ / 2. - v_ / 2. - w_;
	}

	[[nodiscard]] Ray RayThrough(const double s, const double t) const {
		return Ray{origin_, lower_left_corner_ + s * u_ + t * v_ - origin_};
	}

private:
	static constexpr glm::dvec3 kWorldUp{0., 1., 0.};
	glm::dvec3 origin_;
	glm::dvec3 u_{}, v_{}, w_{};
	glm::dvec3 lower_left_corner_{};
};
}
