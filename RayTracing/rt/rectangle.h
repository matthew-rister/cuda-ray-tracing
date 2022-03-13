#pragma once

#include <memory>
#include <optional>

#include "rt/aabb.h"
#include "rt/intersection.h"
#include "rt/material.h"

namespace rt {

class RectangleXY final : public Intersectable {

public:
	RectangleXY(
		const double x0, const double x1,
		const double y0, const double y1,
		const double z,
		std::shared_ptr<Material> material) noexcept
		: x0_{x0}, x1_{x1},
		  y0_{y0}, y1_{y1},
		  z_{z},
		  material_{std::move(material)} {}

	[[nodiscard]] std::optional<Intersection> Intersect(
		const Ray& ray, const double t_min, const double t_max) const override {

		const auto t = (z_ - ray.origin().z) / ray.direction().z;
		if (t < t_min || t_max < t) return std::nullopt;

		const auto point = ray.PointAt(t);
		const auto x = point.x;
		const auto y = point.y;

		if (x < x0_ || x1_ < x || y < y0_ || y1_ < y) return std::nullopt;

		return Intersection{
			.point = point,
			.normal = kNormal,
			.material = material_,
			.t = t,
			.u = (x - x0_) / (x1_ - x0_),
			.v = (y - y0_) / (y1_ - y0_),
			.front_facing = glm::dot(ray.direction(), kNormal) < 0
		};
	}

protected:
	[[nodiscard]] AxisAlignedBoundingBox MakeAxisAlignedBoundingBox() noexcept override {
		return AxisAlignedBoundingBox{glm::dvec3{x0_, y0_, z_ - kEpsilon}, glm::dvec3{x1_, y1_, z_ + kEpsilon}};
	}

private:
	static constexpr glm::dvec3 kNormal{0., 0., 1.};
	static constexpr double kEpsilon = 1e-4;
	double x0_, x1_, y0_, y1_, z_;
	std::shared_ptr<Material> material_;
};

class RectangleXZ final : public Intersectable {

public:
	RectangleXZ(
		const double x0, const double x1,
		const double y,
		const double z0, const double z1,
		std::shared_ptr<Material> material) noexcept
		: x0_{x0}, x1_{x1},
		  y_{y},
		  z0_{z0}, z1_{z1},
		  material_{std::move(material)} {}

	[[nodiscard]] std::optional<Intersection> Intersect(
		const Ray& ray, const double t_min, const double t_max) const override {

		const auto t = (y_ - ray.origin().y) / ray.direction().y;
		if (t < t_min || t_max < t) return std::nullopt;

		const auto point = ray.PointAt(t);
		const auto x = point.x;
		const auto z = point.z;

		if (x < x0_ || x1_ < x || z < z0_ || z1_ < z) return std::nullopt;

		return Intersection{
			.point = point,
			.normal = kNormal,
			.material = material_,
			.t = t,
			.u = (x - x0_) / (x1_ - x0_),
			.v = (z - z0_) / (z1_ - z0_),
			.front_facing = glm::dot(ray.direction(), kNormal) < 0
		};
	}

protected:
	[[nodiscard]] AxisAlignedBoundingBox MakeAxisAlignedBoundingBox() noexcept override {
		return AxisAlignedBoundingBox{glm::dvec3{x0_, y_ - kEpsilon, z0_}, glm::dvec3{x1_, y_ + kEpsilon, z1_}};
	}

private:
	static constexpr glm::dvec3 kNormal{0., 1., 0.};
	static constexpr double kEpsilon = 1e-4;
	double x0_, x1_, y_, z0_, z1_;
	std::shared_ptr<Material> material_;
};

class RectangleYZ final : public Intersectable {

public:
	RectangleYZ(
		const double x,
		const double y0, const double y1,
		const double z0, const double z1,
		std::shared_ptr<Material> material) noexcept
		: x_{x},
		  y0_{y0}, y1_{y1},
		  z0_{z0}, z1_{z1},
		  material_{std::move(material)} {}

	[[nodiscard]] std::optional<Intersection> Intersect(
		const Ray& ray, const double t_min, const double t_max) const override {

		const auto t = (x_ - ray.origin().x) / ray.direction().x;
		if (t < t_min || t_max < t) return std::nullopt;

		const auto point = ray.PointAt(t);
		const auto y = point.y;
		const auto z = point.z;

		if (y < y0_ || y1_ < y || z < z0_ || z1_ < z) return std::nullopt;

		return Intersection{
			.point = point,
			.normal = kNormal,
			.material = material_,
			.t = t,
			.u = (y - y0_) / (y1_ - y0_),
			.v = (z - z0_) / (z1_ - z0_),
			.front_facing = glm::dot(ray.direction(), kNormal) < 0
		};
	}

protected:
	[[nodiscard]] AxisAlignedBoundingBox MakeAxisAlignedBoundingBox() noexcept override {
		return AxisAlignedBoundingBox{glm::dvec3{x_ - kEpsilon, y0_, z0_}, glm::dvec3{x_ + kEpsilon, y1_, z1_}};
	}

private:
	static constexpr glm::dvec3 kNormal{1., 0., 0};
	static constexpr double kEpsilon = 1e-4;
	double x_, y0_, y1_, z0_, z1_;
	std::shared_ptr<Material> material_;
};
}
