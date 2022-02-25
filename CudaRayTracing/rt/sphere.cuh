#pragma once

#include <cmath>

#include <cuda_runtime.h>
#include <glm/vec3.hpp>

#include "rt/intersection.cuh"
#include "rt/ray.cuh"

namespace rt {

class Sphere {

public:
	__host__ __device__ Sphere(const glm::vec3& center, const float radius) noexcept
		: center_{center}, radius_{radius} {}

	__device__ bool Intersect(
		const Ray& ray, const float t_min, const float t_max, Intersection& intersection) const noexcept {

		const auto d = ray.origin() - center_;
		const auto a = glm::dot(ray.direction(), ray.direction());
		const auto half_b = glm::dot(ray.direction(), d);
		const auto c = glm::dot(d, d) - radius_ * radius_;

		const auto discriminant = half_b * half_b - a * c;
		if (discriminant < 0.f) return false;

		const auto sqrt_discriminant = std::sqrtf(discriminant);
		auto t = (-half_b - sqrt_discriminant) / a;
		if (t < t_min || t > t_max) {
			t = (-half_b + sqrt_discriminant) / a;
			if (t < t_min || t > t_max) return false;
		}

		const auto point = ray.PointAt(t);
		const auto normal = (point - center_) / radius_;
		const auto front_facing = glm::dot(ray.direction(), normal) < 0.f;

		intersection.point = point;
		intersection.normal = front_facing ? normal : -normal;
		intersection.t = t;
		intersection.front_facing = front_facing;

		return true;
	}

private:
	glm::vec3 center_;
	float radius_;
};

} // namespace rt
