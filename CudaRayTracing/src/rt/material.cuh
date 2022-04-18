#pragma once

#include <cmath>

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "rt/intersection.cuh"
#include "rt/ray.cuh"

namespace rt {

class Material {

public:
	virtual ~Material() = default;

	__device__ [[nodiscard]] virtual Ray Scatter(
		const Ray& ray, const Intersection& intersection, curandState_t* random_state) const = 0;

protected:
	__device__ glm::vec3 MakeRandomVectorInUnitSphere(curandState_t* const random_state) const {
		glm::vec3 v;
		do {
			const auto x = curand_uniform(random_state);
			const auto y = curand_uniform(random_state);
			const auto z = curand_uniform(random_state);
			v = 2.f * glm::vec3{x, y, z} - glm::vec3{1.f};
		} while (glm::length2(v) > 1.f);
		return glm::normalize(v);
	}
};

class Lambertian final : public Material {

public:
	__device__ explicit Lambertian(const glm::vec3& albedo) noexcept : albedo_{albedo} {}

	__device__ [[nodiscard]] Ray Scatter(
		const Ray& ray, const Intersection& intersection, curandState_t* random_state) const override {
		auto reflection_direction = intersection.normal + MakeRandomVectorInUnitSphere(random_state);
		if (constexpr auto kEpsilon = 1e-9f; glm::length2(reflection_direction) < kEpsilon * kEpsilon) {
			reflection_direction = intersection.normal; // handle case where reflected direction is the zero vector
		}
		return Ray{intersection.point, reflection_direction, ray.Color() * albedo_};
	}

private:
	glm::vec3 albedo_;
};

class Metal final : public Material {

public:
	__device__ explicit Metal(const glm::vec3& albedo, const float fuzz = 0.f) noexcept
		: albedo_{albedo}, fuzz_{fuzz} {}

	__device__ [[nodiscard]] Ray Scatter(
		const Ray& ray, const Intersection& intersection, curandState_t* random_state) const override {
		const auto reflection_direction = glm::reflect(ray.Direction(), intersection.normal);
		const auto fuzz_direction = fuzz_ * MakeRandomVectorInUnitSphere(random_state);
		return Ray{intersection.point, reflection_direction + fuzz_direction, ray.Color() * albedo_};
	}

private:
	glm::vec3 albedo_;
	float fuzz_;
};

class Dielectric final : public Material {

public:
	__device__ explicit Dielectric(const float refractive_index) noexcept : refractive_index_{refractive_index} {}

	__device__ [[nodiscard]] Ray Scatter(
		const Ray& ray, const Intersection& intersection, curandState_t* const random_state) const override {
		const auto refraction_ratio = intersection.front_facing ? 1.f / refractive_index_ : refractive_index_;
		const auto cos_theta = std::fmin(glm::dot(-ray.Direction(), intersection.normal), 1.f);
		const auto direction = CanRefract(cos_theta, refraction_ratio, random_state)
			                       ? glm::refract(ray.Direction(), intersection.normal, refraction_ratio)
			                       : glm::reflect(ray.Direction(), intersection.normal);
		return Ray{intersection.point, direction, ray.Color()};
	}

private:
	__device__ static bool CanRefract(
		const float cos_theta, const float refraction_ratio, curandState_t* const random_state) {

		// verify solution to snell's law exists
		const auto sin_theta = std::sqrtf(1.f - cos_theta * cos_theta);
		if (refraction_ratio * sin_theta > 1.f) return false;

		// schlick's approximation for reflectance
		const auto r0 = std::powf((1.f - refraction_ratio) / (1.f + refraction_ratio), 2.f);
		return r0 + (1.f - r0) * std::powf(1.f - cos_theta, 5.f) < curand_uniform(random_state);
	}

	float refractive_index_;
};
}
