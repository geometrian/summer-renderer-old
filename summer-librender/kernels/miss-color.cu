#pragma once


#include "helpers.cuh"


namespace Summer {


extern "C" __global__ void __miss__color() {
	uint32_t index = optixGetPayload_0();

	interface.camera.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( Vec3f(0.5f), 1.0f ));
}


}
