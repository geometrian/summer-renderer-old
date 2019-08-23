#pragma once


#define OPTIX_COMPATIBILITY 7

#ifdef __INTELLISENSE__
	#define __OPTIX_INCLUDE_INTERNAL_HEADERS__
	#define __forceinline__
	#define __device__
	#include <optix_7_device.h>
#else
	#include <optix_device.h>
#endif

#include "../scene/scenegraph.hpp"

#include "sbt-entries.hpp"


extern "C" {
	__constant__ Summer::Scene::Scene::InterfaceGPU interface;
}
