#pragma once


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


extern "C" {
	__constant__ Summer::Scene::Scene::InterfaceGPU interface;
}
