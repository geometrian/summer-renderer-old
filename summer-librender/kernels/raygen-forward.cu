#pragma once


#include "helpers.cuh"


namespace Summer {


extern "C" __global__ void __raygen__forward() {
	uint3 thread_index = optixGetLaunchIndex();

	uint32_t index = thread_index.y*interface.camera.framebuffer.res[0] + thread_index.x;

	Vec2f res = Vec2f(interface.camera.framebuffer.res);
	Vec2f uv = Vec2f(thread_index.x+0.5f,thread_index.y+0.5f) / res;
	float aspect = res.x / res.y;

	Vec3f const& pos = interface.camera.lookat.position;
	Vec3f const& cen = interface.camera.lookat.center;
	Vec3f        up  = interface.camera.lookat.up;

	Vec3f dir = glm::normalize(cen-pos);
	Vec3f x   = glm::normalize(glm::cross(dir,up));
	up = glm::cross(x,dir);

	Vec3f out = glm::normalize(
		dir + ((uv.x-0.5f)*aspect)*x + (uv.y-0.5f)*up
	);

	optixTrace(
		interface.traversable,

		to_float3(pos), to_float3(out),

		0.0f, std::numeric_limits<float>::infinity(),
		0.0f,

		OptixVisibilityMask(0b11111111),

		OptixRayFlags::OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0u, 0u,
		0u,

		index
	);

	//interface.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( Vec3f(0.5f), 1.0f ));
}


}
