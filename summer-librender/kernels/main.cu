#include "helpers.hpp"


namespace Summer {


/*enum RAY_TYPE {
	PRIMARY = 0,
	SHADOW  = 1
};*/


extern "C" __global__ void __raygen__primary() {
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

		0.0f, 1e20f,//std::numeric_limits<float>::infinity(),
		0.0f,

		OptixVisibilityMask(0b11111111),

		OptixRayFlags::OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0u, 1u,
		0u,

		index
	);

	//interface.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( Vec3f(0.5f), 1.0f ));
}


extern "C" __global__ void __miss__primary() {
	uint32_t index = optixGetPayload_0();
	interface.camera.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( Vec3f(0.9f), 1.0f ));
}

extern "C" __global__ void __miss__shadow () {
	uint32_t index = optixGetPayload_0();
	interface.camera.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( Vec3f(0.2f), 1.0f ));
}


extern "C" __global__ void __closesthit__radiance() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());

	unsigned int prim_index = optixGetPrimitiveIndex();
	Vec3u indices;
	if (data.indices_u16.is_valid()) indices=Vec3u(
		data.indices_u16[3*prim_index  ],
		data.indices_u16[3*prim_index+1],
		data.indices_u16[3*prim_index+2]
	); else if (data.indices_u32.is_valid()) indices=Vec3u(
		data.indices_u32[3*prim_index  ],
		data.indices_u32[3*prim_index+1],
		data.indices_u32[3*prim_index+2]
	); else {
		indices = Vec3u( 3*prim_index, 3*prim_index+1, 3*prim_index+2 );
	}

	float2 bary_uv = optixGetTriangleBarycentrics();
	Vec3f bary = Vec3f( 1.0f-bary_uv.x-bary_uv.y, bary_uv.x, bary_uv.y);
	Vec3f normal0 = data.norms[indices.x];
	Vec3f normal1 = data.norms[indices.y];
	Vec3f normal2 = data.norms[indices.z];
	Vec3f normal = bary.x*normal0 + bary.y*normal1 + bary.z*normal2;

	normal = glm::abs(glm::normalize( normal ));

	uint32_t index = optixGetPayload_0();
	interface.camera.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( normal, 1.0f ));
}

extern "C" __global__ void __anyhit__radiance() {
	uint32_t index = optixGetPayload_0();
	interface.camera.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( 0.5f,0.0f,0.0f, 1.0f ));
}

  
}
