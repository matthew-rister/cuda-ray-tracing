#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>

#include "rt/ray.h"

namespace rt {

class AxisAlignedBoundingBox {

public:
	AxisAlignedBoundingBox() noexcept = default;
	AxisAlignedBoundingBox(const glm::dvec3& p_min, const glm::dvec3& p_max) noexcept : p_min_{p_min}, p_max_{p_max} {}
	AxisAlignedBoundingBox(const AxisAlignedBoundingBox& a, const AxisAlignedBoundingBox& b) noexcept
		: p_min_{glm::min(a.p_min_, b.p_min_)}, p_max_{glm::max(a.p_max_, b.p_max_)} {}

	[[nodiscard]] const glm::dvec3& p_min() const noexcept { return p_min_; }
	[[nodiscard]] const glm::dvec3& p_max() const noexcept { return p_max_; }

	[[nodiscard]] bool Intersect(const Ray& ray, double t_min, double t_max) const {
		const auto direction_inv = 1. / ray.direction();
		const auto t0 = direction_inv * (p_min_ - ray.origin());
		const auto t1 = direction_inv * (p_max_ - ray.origin());
		const auto t0_min = glm::min(t0, t1);
		const auto t1_max = glm::max(t0, t1);
		t_min = std::max(t_min, glm::compMax(t0_min));
		t_max = std::min(t_max, glm::compMin(t1_max));
		return t_min < t_max;
	}

private:
	glm::dvec3 p_min_, p_max_;
};
}
