#pragma once

#include <memory>

#include <cuda_runtime.h>

#include "cuda_error_check.cuh"

template <typename T>
class CudaManaged {

public:
	template <typename ...Args>
	__host__ static std::unique_ptr<T> Make(Args&&... args) {
		return std::make_unique<T>(args...);
	}

	__host__ void* operator new(const size_t size) {
		void* ptr;
		CHECK_CUDA_ERRORS(cudaMallocManaged(&ptr, size));
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
		return ptr;
	}

	__host__ void operator delete(void* ptr) {
		CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
		CHECK_CUDA_ERRORS(cudaFree(ptr));
	}
};