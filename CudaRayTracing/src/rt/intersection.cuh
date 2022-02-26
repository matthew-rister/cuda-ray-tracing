#pragma once

#include <glm/vec3.hpp>

namespace rt {

struct Intersection {
	glm::vec3 point{}, normal{};
	float t{};
	bool front_facing{};
};

} // namespace rt