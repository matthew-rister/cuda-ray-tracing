#pragma once

#include <memory>
#include <variant>

#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include "rt/intersection.h"
#include "rt/ray.h"
#include "rt/texture.h"

namespace rt {

class Material {

public:
	[[nodiscard]] virtual std::optional<Ray> Scatter(const Ray&, const Intersection&) const { return std::nullopt; }
	[[nodiscard]] virtual std::optional<glm::dvec3> Emission() const { return std::nullopt; }

protected:
	~Material() = default;
};

class Lambertian final : public Material {

public:
	explicit constexpr Lambertian(const glm::dvec3& albedo = glm::dvec3{1.}) noexcept : albedo_{albedo} {}

	[[nodiscard]] std::optional<Ray> Scatter(const Ray&, const Intersection& intersection) const override {
		const auto reflection = MakeRandomVectorInUnitHemisphere(intersection.normal);
		return Ray{intersection.point, reflection, albedo_};
	}

private:
	static glm::dvec3 MakeRandomVectorInUnitHemisphere(const glm::dvec3& normal) {
		const auto v = glm::sphericalRand(1.);
		return glm::dot(v, normal) > 0. ? v : -v;
	}

	glm::dvec3 albedo_;
};

class Metal final : public Material {

public:
	explicit constexpr Metal(const glm::dvec3& albedo, const double fuzz = 0.) noexcept : albedo_{albedo}, fuzz_{fuzz} {}

	[[nodiscard]] std::optional<Ray> Scatter(const Ray& incident_ray, const Intersection& intersection) const override {
		const auto reflection = reflect(incident_ray.direction(), intersection.normal);
		const auto reflection_fuzz = fuzz_ * glm::sphericalRand(1.);
		return Ray{intersection.point, reflection + reflection_fuzz, albedo_};
	}

private:
	glm::dvec3 albedo_, fuzz_;
};

class Dielectric final : public Material {

public:
	explicit constexpr Dielectric(const double refractive_index) noexcept : refractive_index_{refractive_index} {}

	[[nodiscard]] std::optional<Ray> Scatter(const Ray& incident_ray, const Intersection& intersection) const override {
		const auto refraction_ratio = intersection.front_facing ? 1. / refractive_index_ : refractive_index_;
		const auto cos_theta = std::min(glm::dot(-incident_ray.direction(), intersection.normal), 1.);

		if (CanRefract(cos_theta, refraction_ratio)) {
			const auto refraction = refract(incident_ray.direction(), intersection.normal, refraction_ratio);
			return Ray{intersection.point, refraction};
		}

		const auto reflection = reflect(incident_ray.direction(), intersection.normal);
		return Ray{intersection.point, reflection};
	}

private:
	static bool CanRefract(const double cos_theta, const double refraction_ratio) {

		// verify solution to snell's law exists
		const auto sin_theta = glm::sqrt(1. - cos_theta * cos_theta);
		if (refraction_ratio * sin_theta > 1.) return false;

		// schlick's approximation for reflectance
		const auto r0 = glm::pow((1. - refraction_ratio) / (1. + refraction_ratio), 2.);
		return r0 + (1. - r0) * glm::pow(1. - cos_theta, 5.) < glm::linearRand(0., 1.);
	}

	double refractive_index_;
};

class TextureMaterial final : public Material {

public:
	TextureMaterial(std::shared_ptr<Material> material, std::shared_ptr<Texture> texture) noexcept
		: material_{std::move(material)}, texture_{std::move(texture)} {}

	[[nodiscard]] std::optional<Ray> Scatter(const Ray& incident_ray, const Intersection& intersection) const override {
		if (const auto ray = material_->Scatter(incident_ray, intersection)) {
			const auto color = texture_->ColorAt(intersection.u, intersection.v, intersection.point);
			return Ray{ray->origin(), ray->direction(), color};
		}
		return std::nullopt;
	}

private:
	std::shared_ptr<Material> material_;
	std::shared_ptr<Texture> texture_;
};

class LambertianEmitter final : public Material {

public:
	explicit LambertianEmitter(const glm::dvec3 emission) noexcept : emission_{emission} {}

	[[nodiscard]] std::optional<glm::dvec3> Emission() const noexcept override { return emission_; }

private:
	glm::dvec3 emission_;
};
}
