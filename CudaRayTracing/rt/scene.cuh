#pragma once

#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_error_check.cuh"
#include "cuda_managed.cuh"
#include "rt/sphere.cuh"

namespace rt {

class Scene final : public CudaManaged<Scene> {

	struct SceneIterator {

		__device__ SceneIterator(const Sphere* objects, const int size, const int index)
			: objects{objects}, size{size}, index{index} {}

		__device__ const Sphere& operator*() const noexcept { return objects[index]; }
		__device__ void operator++() noexcept { ++index; }
		__device__ bool operator!=(const SceneIterator& rhs) const noexcept { return index != rhs.index; }

		const Sphere* objects;
		int size, index;
	};

public:
	__host__ explicit Scene(const std::vector<Sphere>& objects) : size_{static_cast<int>(objects.size())} {
		CHECK_CUDA_ERRORS(cudaMallocManaged(reinterpret_cast<void**>(&objects_), sizeof(Sphere) * size_));
		std::copy(objects.begin(), objects.end(), objects_);
	}

	__host__ ~Scene() { CHECK_CUDA_ERRORS_NOTHROW(cudaFree(objects_)); }

	__device__ [[nodiscard]] SceneIterator begin() const noexcept { return SceneIterator{objects_, size_, 0}; }
	__device__ [[nodiscard]] SceneIterator end() const noexcept { return SceneIterator{objects_, size_, size_}; }

private:
	Sphere* objects_{};
	int size_;
};

} // namespace rt
