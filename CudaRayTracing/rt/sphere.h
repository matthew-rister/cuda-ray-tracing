#pragma once

#include <glm/vec3.hpp>
#include <glm/gtx/norm.hpp>

#include "rt/ray.h"

namespace rt {

class Sphere {

public:
	Sphere(const glm::vec3& center, const float radius) noexcept : center_{center}, radius_{radius} {}

	__device__ [[nodiscard]] bool Intersect(const Ray& ray) const {
		const auto d = ray.origin() - center_;
		const auto a = glm::dot(ray.direction(), ray.direction());
		const auto b = 2.f * glm::dot(ray.direction(), d);
		const auto c = glm::dot(d, d) - radius_ * radius_;
		const auto discriminant = b * b - 4.f * a * c;
		return discriminant > 0.f;
	}

private:
	glm::vec3 center_;
	float radius_;
};

}
