#pragma once

#include <cuda_runtime_api.h>

#include "rt/material.cuh"

namespace rt {
struct Intersection;
class Ray;

class Hittable {

public:
	__device__ explicit Hittable(const Material* const material) noexcept : material_{material} {}
	virtual ~Hittable() { delete material_; }

	__device__ [[nodiscard]] const Material* material() const noexcept { return material_; }
	__device__ [[nodiscard]] virtual Intersection Intersect(const Ray& ray, float t_min, float t_max) const = 0;

protected:
	const Material* material_;
};
}
