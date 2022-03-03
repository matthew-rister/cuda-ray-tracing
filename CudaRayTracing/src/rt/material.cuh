#pragma once

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>

#include "rt/ray.cuh"

namespace rt {

class Material {

public:
	__device__ virtual ~Material() {}

	__device__ [[nodiscard]] virtual Ray Scatter(
		const Ray& ray, const glm::vec3& point, const glm::vec3& normal, curandState_t* random_state) const = 0;

protected:
	__device__ glm::vec3 MakeRandomVectorInUnitSphere(curandState_t* random_state) const {
		glm::vec3 v;
		do {
			const auto x = curand_uniform(random_state);
			const auto y = curand_uniform(random_state);
			const auto z = curand_uniform(random_state);
			v = 2.f * glm::vec3{ x, y, z } - glm::vec3{ 1.f };
		} while (glm::dot(v, v) > 1.f);
		return glm::normalize(v);
	}
};

class Lambertian final : public Material {

public:
	__device__ explicit Lambertian(const glm::vec3& albedo) noexcept : albedo_{albedo} {}

	__device__ [[nodiscard]] Ray Scatter(
		const Ray& ray, const glm::vec3& point, const glm::vec3& normal, curandState_t* random_state) const override {
		auto reflection = normal + MakeRandomVectorInUnitSphere(random_state);
		if (constexpr auto kEpsilon = 1e-9f; glm::dot(reflection, reflection) < kEpsilon * kEpsilon) {
			reflection = normal; // handle case where reflection is opposite normal resulting in the zero vector
		}
		return Ray{point, reflection, ray.color() * albedo_};
	}

private:
	glm::vec3 albedo_;
};

class Metal final : public Material {

public:
	__device__ explicit Metal(const glm::vec3& albedo, const float fuzz = 0.f) noexcept
		: albedo_{albedo}, fuzz_{fuzz} {}

	__device__ [[nodiscard]] Ray Scatter(
		const Ray& ray, const glm::vec3& point, const glm::vec3& normal, curandState_t* random_state) const override {
		const auto reflection = glm::reflect(ray.direction(), normal);
		const auto fuzz_direction = fuzz_ * MakeRandomVectorInUnitSphere(random_state);
		return Ray{point, reflection + fuzz_direction, ray.color() * albedo_};
	}

private:
	glm::vec3 albedo_;
	float fuzz_;
};

} // namespace rt
