#pragma once

#include <glm/glm.hpp>

namespace rt {

struct Intersection {
	glm::vec3 point, normal;
	float t;
	bool front_facing, hit = false;
};
}
