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
};

class Lambertian final : public Material {

public:
	__device__ explicit Lambertian(const glm::vec3& albedo) noexcept : albedo_{albedo} {}

	__device__ [[nodiscard]] Ray Scatter(
		const Ray& ray, const glm::vec3& point, const glm::vec3& normal, curandState_t* random_state) const override {
		return Ray{point, normal + MakeRandomVectorInUnitSphere(random_state), albedo_ * ray.color()};
	}

private:
	__device__ glm::vec3 MakeRandomVectorInUnitSphere(curandState_t* random_state) const {
		glm::vec3 v;
		do {
			const auto x = curand_uniform(random_state);
			const auto y = curand_uniform(random_state);
			const auto z = curand_uniform(random_state);
			v = 2.f * glm::vec3{x, y, z} - glm::vec3{1.f};
		} while (glm::dot(v, v) > 1.f);
		return glm::normalize(v);
	}

	glm::vec3 albedo_;
};

} // namespace rt
