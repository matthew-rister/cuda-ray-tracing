#pragma once

#include <cmath>

#include <cuda_runtime_api.h>
#include <glm/vec3.hpp>

#include "rt/intersection.cuh"
#include "rt/material.cuh"
#include "rt/ray.cuh"

namespace rt {

class Sphere final : public Intersectable {

public:
	__device__ Sphere(const glm::vec3& center, const float radius, Material* material) noexcept
		: center_{center}, radius_{radius}, material_{material} {}

	__device__ ~Sphere() override { delete material_; }

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
		return {point, front_facing ? normal : -normal, t, front_facing, true, material_};
	}

private:
	glm::vec3 center_;
	float radius_;
	Material* material_;
};

} // namespace rt
