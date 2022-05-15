# CUDA Ray Tracing

This project implements a CUDA accelerated version of Peter Shirley's 
[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) with a focus on writing 
clean, maintainable code using modern C++ features where possible. For reference, each chapter's implementation is 
tracked using separate git branches. Additionally, a non-CUDA implementation for the next part in the series, 
[Ray Tracing The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html) can be found 
[here](https://github.com/matthew-rister/cuda-ray-tracing/tree/matthew-rister/rtnextweek).

![](https://github.com/matthew-rister/cuda-ray-tracing/blob/main/CudaRayTracing/img/ch13.png)

### Performance

In contrast to the original implementation, which can take several hours to finish rendering the final scene, this CUDA
implementation finishes in about 7 minutes using an NVIDIA 3080 Ti. Note this is without the use of 
additional acceleration structures which could be used to further reduce the amount of time needed to render a scene.

## How to Run

This project was built using Visual Studio, CUDA 11.6, and C++ 17 with code generation targeting NVIDIA's Ampere 
Architecture. Additionally, this project utilizes [GLM](https://github.com/g-truc/glm) and 
[STB Image](https://github.com/nothings/stb) libraries which are managed using [vcpkg](https://vcpkg.io/en/index.html) 
and configured to install on first build. 
