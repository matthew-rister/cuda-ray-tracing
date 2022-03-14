#pragma once

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#ifdef NDEBUG
	#define CHECK_CUDA_ERRORS(status) status
	#define CHECK_CUDA_ERRORS_NOTHROW(status) status
#else
	#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
	#define CHECK_CUDA_ERRORS(status) rt::CheckCudaErrors((status), #status, __FILENAME__, __LINE__, true)
	#define CHECK_CUDA_ERRORS_NOTHROW(status) rt::CheckCudaErrors((status), #status, __FILENAME__, __LINE__, false)
#endif

namespace rt {

inline void CheckCudaErrors(
	const cudaError_t status, const char* function, const char* filename, const int line_number, const bool throws) {

	if (status != cudaSuccess) {
		const auto* error = cudaGetErrorString(status);
		std::ostringstream oss;
		oss << function << " failed at " << filename << ':' << line_number << " with error \"" << error << '\"';
		if (throws) throw std::runtime_error{oss.str()};
		std::cerr << oss.str() << '\n';
	}
}

}