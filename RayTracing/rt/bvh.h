#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "rt/aabb.h"
#include "rt/intersection.h"
#include "rt/ray.h"

namespace rt {

class BoundingVolumeHierarchy {

	struct BoundingVolumeHierarchyNode final : Intersectable {

		BoundingVolumeHierarchyNode(
			std::vector<std::shared_ptr<Intersectable>>& objects, const int begin, const int end) {

			static auto axis = 0;
			if (++axis > 2) axis = 0;

			if (const auto size = end - begin; size == 1) {
				left_child = objects.at(begin);
			} else {
				std::sort(objects.begin() + begin, objects.begin() + end, [](auto& lhs, auto& rhs) {
					return lhs->aabb().p_min()[axis] < rhs->aabb().p_min()[axis];
				});
				const auto mid = begin + size / 2;
				left_child = std::make_shared<BoundingVolumeHierarchyNode>(objects, begin, mid);
				right_child = std::make_shared<BoundingVolumeHierarchyNode>(objects, mid, end);
			}

			aabb_ = MakeAxisAlignedBoundingBox();
		}

		[[nodiscard]] std::optional<Intersection> Intersect(
			const Ray& ray, const double t_min, double t_max) const override {

			if (!aabb_->Intersect(ray, t_min, t_max)) return std::nullopt;

			const auto left_intersection = left_child->Intersect(ray, t_min, t_max);
			if (left_intersection) t_max = left_intersection->t;

			const auto right_intersection = right_child ? right_child->Intersect(ray, t_min, t_max) : std::nullopt;
			return right_intersection ? right_intersection : left_intersection;
		}

		std::shared_ptr<Intersectable> left_child, right_child;

	protected:
		[[nodiscard]] AxisAlignedBoundingBox MakeAxisAlignedBoundingBox() override {
			if (right_child) {
				return AxisAlignedBoundingBox{left_child->aabb(), right_child->aabb()};
			}
			return left_child->aabb();
		}
	};

public:
	explicit BoundingVolumeHierarchy(std::vector<std::shared_ptr<Intersectable>> objects)
		: root_{std::make_unique<BoundingVolumeHierarchyNode>(objects, 0, static_cast<int>(objects.size()))} {}

	[[nodiscard]] std::optional<Intersection> Intersect(const Ray& ray, const double t_min, const double t_max) const {
		return root_->Intersect(ray, t_min, t_max);
	}

private:
	std::unique_ptr<BoundingVolumeHierarchyNode> root_;
};
}
