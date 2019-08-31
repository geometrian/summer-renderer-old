#pragma once


#include "generic-forward.cu"


namespace Summer {


class TraceInfoAmbientOcclusion final {
	public:
		RNG* rng;

		float visibility;
};


extern "C" __global__ void __raygen__ambientocclusion() {
	generic_forward0_raygen();
}


extern "C" __global__ void __miss__ambientocclusion_normal() {
	TraceInfoBasic const* info = generic_forward0_miss();

	semiAtomicAdd(interface.camera.framebuffer.layers.lighting_integration+info->index.pixel_flat,Vec4f( Vec3f(1.0f), 1.0f ));
}
extern "C" __global__ void __miss__ambientocclusion_shadow() {
	TraceInfoAmbientOcclusion* info_shad = PackedPointer<TraceInfoAmbientOcclusion>::from_payloads01();

	info_shad->visibility = 1.0f;
}


extern "C" __global__ void __anyhit__ambientocclusion_normal() {
	generic_forward0_anyhit();
}
extern "C" __global__ void __anyhit__ambientocclusion_shadow() {
	TraceInfoAmbientOcclusion* info_shad = PackedPointer<TraceInfoAmbientOcclusion>::from_payloads01();

	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	Vec4f albedo = shade_point.material->get_albedo(&shade_point);
	if        (albedo.a==1.0f) {
		info_shad->visibility = 0.0f;
	} else if (albedo.a!=0.0f) {
		if (info_shad->rng->get_uniform() <= albedo.a) {
			info_shad->visibility = 0.0f;
		} else {
			optixIgnoreIntersection();
		}
	} else {
		optixIgnoreIntersection();
	}
}


extern "C" __global__ void __closesthit__ambientocclusion_normal() {
	TraceInfoBasic const* info = generic_forward0_closesthit();

	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	TraceInfoAmbientOcclusion info_shad = { info->rng, 1.0f };

	Ray ray_shad = { shade_point.pos, info->rng->get_coshemi(shade_point.Nshad) };
	offset_ray_orig( &ray_shad, shade_point.Ngeom );

	PackedPointer<TraceInfoAmbientOcclusion> ptr = &info_shad;
	optixTrace(
		interface.traversable,

		to_float3(ray_shad.orig), to_float3(ray_shad.dir),

		0.0f, 4.0f,//std::numeric_limits<float>::infinity(),
		0.0f,

		OptixVisibilityMask(0b11111111),

		OptixRayFlags::OPTIX_RAY_FLAG_NONE,
		1u, 2u,
		1u,

		ptr[0], ptr[1]
	);

	Vec4f color = Vec4f(Vec3f(info_shad.visibility),1.0f);
	semiAtomicAdd(interface.camera.framebuffer.layers.lighting_integration+info->index.pixel_flat,color);
}

  
}
