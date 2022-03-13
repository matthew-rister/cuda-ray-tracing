#pragma once

#include <optional>

#include <glm/glm.hpp>

namespace rt {
class AxisAlignedBoundingBox;
class Material;
class Ray;

struct Intersection {
	glm::dvec3 point;
	glm::dvec3 normal;
	std::shared_ptr<Material> material;
	double t;
	double u, v;
	bool front_facing;
};

class Intersectable {

public:
	[[nodiscard]] virtual std::optional<Intersection> Intersect(const Ray& ray, double t_min, double t_max) const = 0;
	[[nodiscard]] const AxisAlignedBoundingBox& aabb() {
		if (!aabb_) aabb_ = MakeAxisAlignedBoundingBox();
		return *aabb_;
	}

protected:
	~Intersectable() = default;
	[[nodiscard]] virtual AxisAlignedBoundingBox MakeAxisAlignedBoundingBox() = 0;
	std::optional<AxisAlignedBoundingBox> aabb_;
};
}
