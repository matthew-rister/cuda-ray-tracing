#pragma once

#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

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
	virtual ~Intersectable() = default;

	__device__ [[nodiscard]] virtual Intersection Intersect(const Ray& ray, float t_min, float t_max) const = 0;
};

}
