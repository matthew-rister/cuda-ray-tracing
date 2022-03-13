#pragma once

#include <cmath>

#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

#include "rt/hittable.cuh"
#include "rt/intersection.cuh"
#include "rt/material.cuh"
#include "rt/ray.cuh"

namespace rt {

class Sphere final : public Hittable {

public:
	__device__ Sphere(const glm::vec3& center, const float radius, const Material* material) noexcept
		: Hittable{material}, center_{center}, radius_{radius} {}

	__device__ [[nodiscard]] Intersection Intersect(
		const Ray& ray, const float t_min, const float t_max) const override {

		const auto d = ray.origin() - center_;
		const auto a = glm::dot(ray.direction(), ray.direction());
		const auto half_b = glm::dot(ray.direction(), d);
		const auto c = glm::dot(d, d) - radius_ * radius_;

		const auto discriminant = half_b * half_b - a * c;
		if (discriminant < 0.f) return {};

		const auto sqrt_discriminant = std::sqrtf(discriminant);
		auto t = (-half_b - sqrt_discriminant) / a;
		if (t < t_min || t > t_max) {
			t = (-half_b + sqrt_discriminant) / a;
			if (t < t_min || t > t_max) return {};
		}

		const auto point = ray.PointAt(t);
		const auto normal = (point - center_) / radius_;
		const auto front_facing = glm::dot(ray.direction(), normal) < 0.f;
		return {point, front_facing ? normal : -normal, t, front_facing, true};
	}

private:
	glm::vec3 center_;
	float radius_;
};

}
