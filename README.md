# CUDA Ray Tracing

This project provides a CUDA accelerated implementation of Peter Shirley's 
[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) with a focus on writing 
clean, maintainable code using modern C++ features where possible. For reference, each chapter's implementation is 
tracing using dedicated git branches which can be found [here](https://github.com/matthew-rister/cuda-ray-tracing/branches). 

![](https://github.com/matthew-rister/cuda-ray-tracing/blob/main/CudaRayTracing/img/ch13.png)

## How to Run

This project was built in Visual Studio using CUDA 11.6 and C++17 with code generation targeting NVIDIA's Ampere 
Architecture. Additionally, this project utilizes the [GLM](https://github.com/g-truc/glm) 
and [STB](https://github.com/nothings/stb) libraries which are managed using [vcpkg](https://vcpkg.io/en/index.html) and 
configured to install on first build. 