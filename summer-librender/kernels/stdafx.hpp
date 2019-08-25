#pragma once


#define OPTIX_COMPATIBILITY 7

#ifdef __INTELLISENSE__
	#define __OPTIX_INCLUDE_INTERNAL_HEADERS__
	#define __CUDACC__
	#include <optix_7_device.h>
	#undef __CUDACC__
	#undef __constant__
	#undef __device__
	#undef __forceinline__
	#undef __global__
	#define __constant__
	#define __device__
	#define __forceinline__
	#define __global__
#else
	#include <optix_device.h>
#endif
//#include <cuda_device_runtime_api.h>

#include "../scene/materials/material.hpp"

#include "../scene/scenegraph.hpp"

#include "sbt-entries.hpp"


#define PI       3.14159265358979323846264338327950288419716939937510582097f
#define RECIP_PI 0.318309886183790671537767526745028724068919291480912897495f


extern "C" {
	__constant__ Summer::Scene::Scene::InterfaceGPU interface;
}
