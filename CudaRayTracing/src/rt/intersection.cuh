#pragma once

#include <cuda_runtime_api.h>
#include <glm/vec3.hpp>

namespace rt {
class Material;
class Ray;

struct Intersection {
	glm::vec3 point, normal;
	float t;
	bool front_facing, hit = false;
	Material* material;
};

class Intersectable {

public:
	__device__ virtual ~Intersectable() {}
	__device__ [[nodiscard]] virtual Intersection Intersect(const Ray& ray, float t_min, float t_max) const = 0;
};

} // namespace rt
