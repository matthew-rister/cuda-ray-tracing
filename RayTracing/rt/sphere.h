#pragma once

#include <memory>
#include <numbers>
#include <optional>

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "rt/intersection.h"
#include "rt/material.h"
#include "rt/texture.h"

namespace rt {

class Sphere final : public Intersectable, public Texturable {

public:
	Sphere(const glm::dvec3& center, const double radius, std::shared_ptr<Material> material) noexcept
		: center_{center}, radius_{radius}, material_{std::move(material)} {}

	[[nodiscard]] std::optional<Intersection> Intersect(
		const Ray& ray, const double t_min, const double t_max) const override {

		const auto delta = ray.origin() - center_;
		const auto a = glm::length2(ray.direction());
		const auto half_b = glm::dot(ray.direction(), delta);
		const auto c = glm::length2(delta) - radius_ * radius_;

		const auto discriminant = pow(half_b, 2) - a * c;
		if (discriminant < 0.) return std::nullopt;

		const auto sqrt_discriminant = glm::sqrt(discriminant);
		auto t = (-half_b - sqrt_discriminant) / a;

		if (t < t_min || t_max < t) {
			t = (-half_b + sqrt_discriminant) / a;
			if (t < t_min || t_max < t) return std::nullopt;
		}

		const auto point = ray.PointAt(t);
		const auto normal = (point - center_) / radius_;
		const auto front_facing = glm::dot(ray.direction(), normal) < 0;
		const auto texture_coordinates = TextureCoordinatesAt(normal);

		return Intersection{
			.point = point,
			.normal = front_facing ? normal : -normal,
			.material = material_,
			.t = t,
			.u = texture_coordinates.s,
			.v = texture_coordinates.t,
			.front_facing = front_facing
		};
	}

	[[nodiscard]] glm::dvec2 TextureCoordinatesAt(const glm::dvec3& normal) const noexcept override {
		const auto theta = acos(-normal.y);
		const auto phi = atan2(-normal.z, normal.x) + kPi;
		const auto u = phi / (2 * kPi);
		const auto v = theta / kPi;
		return {u, v};
	}

protected:
	[[nodiscard]] AxisAlignedBoundingBox MakeAxisAlignedBoundingBox() noexcept override {
		const auto radius = std::abs(radius_); // use std::abs here in case radius is negative due to the hollow bubble trick
		return AxisAlignedBoundingBox{center_ - glm::dvec3{radius}, center_ + glm::dvec3{radius}};
	}

private:
	static constexpr double kPi = std::numbers::pi;
	glm::dvec3 center_;
	double radius_;
	std::shared_ptr<Material> material_;
};
}
