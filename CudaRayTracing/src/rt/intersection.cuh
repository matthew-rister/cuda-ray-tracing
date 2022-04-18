#pragma once

#include <glm/glm.hpp>

namespace rt {
class Material;

struct Intersection {
	glm::vec3 point{}, normal{};
	float t{};
	bool front_facing{}, hit{};
	const Material* material{};
};

class Intersectable {

public:
	__device__ explicit Intersectable() noexcept {}
	__device__ virtual ~Intersectable() {}

	__device__ [[nodiscard]] virtual Intersection Intersect(const Ray& ray, float t_min, float t_max) const = 0;
};
}
