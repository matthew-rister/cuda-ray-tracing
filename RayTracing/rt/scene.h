#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include <glm/glm.hpp>

#include "rectangle.h"
#include "rt/camera.h"
#include "rt/image.h"
#include "rt/intersection.h"
#include "rt/sphere.h"

namespace rt {

struct Scene {
	enum class Type { kThreeSpheres, kRandomScene, kCheckerboardTexture, kPerlinSpheres, kEarthTexture, kSimpleLights };

	static Scene MakeScene(const Type type) {
		switch (type) {
			case Type::kThreeSpheres:
				return MakeThreeSpheres();
			case Type::kRandomScene:
				return MakeRandomScene();
			case Type::kCheckerboardTexture:
				return MakeCheckerboardTexture();
			case Type::kPerlinSpheres:
				return MakePerlinSpheres();
			case Type::kEarthTexture:
				return MakeEarthTexture();
			case Type::kSimpleLights:
				return MakeSimpleLights();
			default:
				throw std::runtime_error{"Unhandled scene type"};
		}
	}

	Image<> image;
	Camera camera;
	int samples_per_pixel;
	int max_depth;
	glm::dvec3 background_color;
	std::vector<std::shared_ptr<Intersectable>> objects;

private:
	static Scene MakeThreeSpheres() {

		constexpr auto kAspectRatio = 16. / 9.;
		constexpr auto kImageWidth = 400;
		constexpr auto kImageHeight = static_cast<int>(kImageWidth / kAspectRatio);
		constexpr auto kSamplesPerPixel = 100;
		constexpr auto kMaxRecursionDepth = 50;
		constexpr glm::dvec3 kBackgroundColor{.7, .8, 1.};
		Image image{kImageWidth, kImageHeight};

		constexpr glm::dvec3 kLookFrom{-2., 2., 1.};
		constexpr glm::dvec3 kLookAt{0., 0., -1.};
		constexpr auto kVerticalFieldOfView = 50.;
		Camera camera{kLookFrom, kLookAt, kAspectRatio, kVerticalFieldOfView};

		std::vector<std::shared_ptr<Intersectable>> objects{
			std::make_shared<Sphere>(glm::dvec3{0., -100.5, -1.}, 100., std::make_shared<Lambertian>(glm::dvec3{.8, .8, 0.})),
			std::make_shared<Sphere>(glm::dvec3{0., 0., -1.}, .5, std::make_shared<Lambertian>(glm::dvec3{.1, .2, .5})),
			std::make_shared<Sphere>(glm::dvec3{-1., 0., -1.}, .5, std::make_shared<Dielectric>(1.5)),
			std::make_shared<Sphere>(glm::dvec3{-1., 0., -1.}, -.45, std::make_shared<Dielectric>(1.5)),
			std::make_shared<Sphere>(glm::dvec3{1., 0., -1.}, .5, std::make_shared<Metal>(glm::dvec3{.8, .6, .2})),
		};

		return Scene{
			.image = std::move(image),
			.camera = std::move(camera),
			.samples_per_pixel = kSamplesPerPixel,
			.max_depth = kMaxRecursionDepth,
			.background_color = kBackgroundColor,
			.objects = std::move(objects)
		};
	}

	static Scene MakeRandomScene() {

		constexpr auto kAspectRatio = 3. / 2.;
		constexpr auto kImageWidth = 400;
		constexpr auto kImageHeight = static_cast<int>(kImageWidth / kAspectRatio);
		constexpr auto kSamplesPerPixel = 100;
		constexpr auto kMaxRecursionDepth = 50;
		constexpr glm::dvec3 kBackgroundColor{.7, .8, 1.};
		Image image{kImageWidth, kImageHeight};

		constexpr glm::dvec3 kLookFrom{13., 2., 3.};
		constexpr glm::dvec3 kLookAt{0.};
		constexpr auto kVerticalFieldOfView = 20.;
		Camera camera{kLookFrom, kLookAt, kAspectRatio, kVerticalFieldOfView};

		std::vector<std::shared_ptr<Intersectable>> objects{
			std::make_shared<Sphere>(glm::dvec3{0., -1000., 0.}, 1000., std::make_shared<Lambertian>(glm::dvec3{.5})),
			std::make_shared<Sphere>(glm::dvec3{0., 1., 0.}, 1., std::make_shared<Dielectric>(1.5)),
			std::make_shared<Sphere>(glm::dvec3{-4., 1., 0.}, 1., std::make_shared<Lambertian>(glm::dvec3{.4, .2, .1})),
			std::make_shared<Sphere>(glm::dvec3{4., 1., 0.}, 1., std::make_shared<Metal>(glm::dvec3{.7, .6, .5}))
		};

		constexpr auto make_random_vector = [](const double min, const double max) {
			return glm::dvec3{glm::linearRand(min, max), glm::linearRand(min, max), glm::linearRand(min, max)};
		};

		constexpr auto half_n = 11;
		for (auto i = -half_n; i < half_n; i++) {
			for (auto j = -half_n; j < half_n; j++) {
				const glm::dvec3 center{i + .9 * glm::linearRand(0., 1.), .2, j + .9 * glm::linearRand(0., 1.)};

				if (glm::length(center - glm::dvec3(4., .2, 0.)) > .9) {
					if (const auto material_probability = glm::linearRand(0., 1.); material_probability < .8) {
						const auto albedo = make_random_vector(0., 1.) * make_random_vector(0., 1.);
						objects.push_back(std::make_shared<Sphere>(center, .2, std::make_shared<Lambertian>(albedo)));
					} else if (material_probability < .95) {
						const auto albedo = make_random_vector(.5, 1.);
						const auto fuzz = glm::linearRand(0., .5);
						objects.push_back(std::make_shared<Sphere>(center, .2, std::make_shared<Metal>(albedo, fuzz)));
					} else {
						objects.push_back(std::make_shared<Sphere>(center, .2, std::make_shared<Dielectric>(1.5)));
					}
				}

			}
		}

		return Scene{
			.image = std::move(image),
			.camera = std::move(camera),
			.samples_per_pixel = kSamplesPerPixel,
			.max_depth = kMaxRecursionDepth,
			.background_color = kBackgroundColor,
			.objects = std::move(objects)
		};
	}

	static Scene MakeCheckerboardTexture() {

		constexpr auto kAspectRatio = 3. / 2.;
		constexpr auto kImageWidth = 400;
		constexpr auto kImageHeight = static_cast<int>(kImageWidth / kAspectRatio);
		constexpr auto kSamplesPerPixel = 100;
		constexpr auto kMaxDepth = 50;
		constexpr glm::dvec3 kBackgroundColor{.7, .8, 1.};
		Image image{kImageWidth, kImageHeight};

		constexpr glm::dvec3 kLookFrom{13., 2., 3.};
		constexpr glm::dvec3 kLookAt{0.};
		constexpr auto kVerticalFieldOfView = 30.;
		Camera camera{kLookFrom, kLookAt, kAspectRatio, kVerticalFieldOfView};

		const auto texture = std::make_shared<Checkerboard>(50, 50, glm::dvec3{.9}, glm::dvec3{.2, .3, .1});
		const auto material = std::make_shared<TextureMaterial>(std::make_shared<Lambertian>(), texture);

		std::vector<std::shared_ptr<Intersectable>> objects{
			std::make_shared<Sphere>(glm::dvec3{0., -10., 0.}, 10., material),
			std::make_shared<Sphere>(glm::dvec3{0., 10., 0.}, 10., material)
		};

		return Scene{
			.image = std::move(image),
			.camera = std::move(camera),
			.samples_per_pixel = kSamplesPerPixel,
			.max_depth = kMaxDepth,
			.background_color = kBackgroundColor,
			.objects = std::move(objects)
		};
	}

	static Scene MakePerlinSpheres() {

		constexpr auto kAspectRatio = 3. / 2.;
		constexpr auto kImageWidth = 400;
		constexpr auto kImageHeight = static_cast<int>(kImageWidth / kAspectRatio);
		constexpr auto kSamplesPerPixel = 100;
		constexpr auto kMaxDepth = 50;
		constexpr glm::dvec3 kBackgroundColor{.7, .8, 1.};
		Image image{kImageWidth, kImageHeight};

		constexpr glm::dvec3 kLookFrom{13., 2., 3.};
		constexpr glm::dvec3 kLookAt{0.};
		constexpr auto kVerticalFieldOfView = 20.;
		Camera camera{kLookFrom, kLookAt, kAspectRatio, kVerticalFieldOfView};

		const auto perlin_texture = std::make_shared<PerlinNoise>(4.);
		const auto perlin_material = std::make_shared<TextureMaterial>(std::make_shared<Lambertian>(), perlin_texture);

		std::vector<std::shared_ptr<Intersectable>> objects{
			std::make_shared<Sphere>(glm::dvec3{0., -1000., 0.}, 1000., perlin_material),
			std::make_shared<Sphere>(glm::dvec3{0., 2., 0.}, 2., perlin_material)
		};

		return Scene{
			.image = std::move(image),
			.camera = std::move(camera),
			.samples_per_pixel = kSamplesPerPixel,
			.max_depth = kMaxDepth,
			.background_color = kBackgroundColor,
			.objects = std::move(objects)
		};
	}

	static Scene MakeEarthTexture() {

		constexpr auto kAspectRatio = 3. / 2.;
		constexpr auto kImageWidth = 400;
		constexpr auto kImageHeight = static_cast<int>(kImageWidth / kAspectRatio);
		constexpr auto kSamplesPerPixel = 100;
		constexpr auto kMaxDepth = 50;
		constexpr glm::dvec3 kBackgroundColor{.7, .8, 1.};
		Image image{kImageWidth, kImageHeight};

		constexpr glm::dvec3 kLookFrom{13., 2., 3.};
		constexpr glm::dvec3 kLookAt{0.};
		constexpr auto kVerticalFieldOfView = 20.;
		Camera camera{kLookFrom, kLookAt, kAspectRatio, kVerticalFieldOfView};

		const auto earth_texture = std::make_shared<Image2d<>>("assets/textures/earth.jpg");
		const auto earth_material = std::make_shared<TextureMaterial>(std::make_shared<Lambertian>(), earth_texture);
		const auto earth_sphere = std::make_shared<Sphere>(glm::dvec3{0.}, 2., earth_material);
		std::vector<std::shared_ptr<Intersectable>> objects{earth_sphere};

		return Scene{
			.image = std::move(image),
			.camera = std::move(camera),
			.samples_per_pixel = kSamplesPerPixel,
			.max_depth = kMaxDepth,
			.background_color = kBackgroundColor,
			.objects = std::move(objects)
		};
	}

	static Scene MakeSimpleLights() {

		constexpr auto kAspectRatio = 16. / 9.;
		constexpr auto kImageWidth = 1440;
		constexpr auto kImageHeight = static_cast<int>(kImageWidth / kAspectRatio);
		constexpr auto kSamplesPerPixel = 500;
		constexpr auto kMaxDepth = 50;
		constexpr glm::dvec3 kBackgroundColor{0.};
		Image image{kImageWidth, kImageHeight};

		constexpr glm::dvec3 kLookFrom{23., 2., 3.};
		constexpr glm::dvec3 kLookAt{0., 2., 0.};
		constexpr auto kVerticalFieldOfView = 20.;
		Camera camera{kLookFrom, kLookAt, kAspectRatio, kVerticalFieldOfView};

		const auto perlin_texture = std::make_shared<PerlinNoise>(4.);
		const auto perlin_material = std::make_shared<TextureMaterial>(std::make_shared<Lambertian>(), perlin_texture);

		std::vector<std::shared_ptr<Intersectable>> objects{
			std::make_shared<Sphere>(glm::dvec3{0., -1000., 0.}, 1000., perlin_material),
			std::make_shared<Sphere>(glm::dvec3{0., 2., 0.}, 2., perlin_material),
			std::make_shared<Sphere>(glm::dvec3{0., 7., 0.}, 2., std::make_shared<LambertianEmitter>(glm::dvec3{4.})),
			std::make_shared<RectangleXY>(3., 5., 1., 3., -2., std::make_shared<LambertianEmitter>(glm::dvec3{4.}))
		};

		return Scene{
			.image = std::move(image),
			.camera = std::move(camera),
			.samples_per_pixel = kSamplesPerPixel,
			.max_depth = kMaxDepth,
			.background_color = kBackgroundColor,
			.objects = std::move(objects)
		};
	}
};
}
