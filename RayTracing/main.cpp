#include <chrono>
#include <format>
#include <iostream>
#include <optional>

#include <glm/glm.hpp>

#include "rt/bvh.h"
#include "rt/image.h"
#include "rt/intersection.h"
#include "rt/ray.h"
#include "rt/scene.h"
#include "rt/sphere.h"

using namespace glm;
using namespace rt;
using namespace std;

dvec3 ComputeRayColor(
	const Ray& ray, const BoundingVolumeHierarchy& bvh, const dvec3& background_color, const int depth) {

	if (depth < 0) return dvec3{0.};

	if (const auto intersection = bvh.Intersect(ray, .001, numeric_limits<double>::infinity())) {
		const auto& material = intersection->material;
		if (const auto emission = material->Emission()) {
			return *emission;
		}
		if (const auto reflection = material->Scatter(ray, *intersection)) {
			return reflection->color() * ComputeRayColor(*reflection, bvh, background_color, depth - 1);
		}
	}

	return background_color;
}

int main() {
	const auto start_time = chrono::high_resolution_clock::now();
	auto [
		image,
		camera,
		samples_per_pixel,
		max_depth,
		background_color,
		objects
	] = Scene::MakeScene(Scene::Type::kSimpleLights);
	const BoundingVolumeHierarchy bvh{objects};

	for (auto i = 0; i < image.height(); ++i) {
		cout << format("{} scanlines remaining...\n", image.height() - i);

		for (auto j = 0; j < image.width(); ++j) {
			dvec3 accumulated_color{0.};

			for (auto k = 0; k < samples_per_pixel; ++k) {
				const auto u = (j + linearRand(0., 1.)) / (image.width() - 1.);
				const auto v = (i + linearRand(0., 1.)) / (image.height() - 1.);
				const auto ray = camera.RayThrough(u, v);
				accumulated_color += ComputeRayColor(ray, bvh, background_color, max_depth);
			}

			const auto average_color = accumulated_color / static_cast<double>(samples_per_pixel);
			const auto gamma_correction = glm::sqrt(average_color);
			image(i, j) = static_cast<double>(image.kMaxColorValue) * gamma_correction;
		}
	}

	image.SaveAs("output.png");

	const auto end_time = chrono::high_resolution_clock::now();
	cout << "Image rendered in " << chrono::duration<double>{end_time - start_time}.count() << " seconds\n";

	return EXIT_SUCCESS;
}
