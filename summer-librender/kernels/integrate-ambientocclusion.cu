#pragma once


#include "generic-forward.cu"


namespace Summer {


class TraceInfoAmbientOcclusion final {
	public:
		RNG*const rng;

		float visibility;

	public:
		__device__ explicit TraceInfoAmbientOcclusion(RNG* rng) : rng(rng) {}
};


extern "C" __global__ void __raygen__ambientocclusion() {
	TraceInfoBasic trace_info;
	generic_forward0_raygen<TraceInfoBasic>(trace_info);
}


extern "C" __global__ void __miss__ambientocclusion_normal() {
	TraceInfoBasic const* trace_info = generic_forward0_miss<TraceInfoBasic>();

	semiAtomicAdd(
		interface.camera.framebuffer.layers.lighting_integration + trace_info->index.pixel_flat,
		Vec4f( Vec3f(1.0f), 1.0f )
	);
}
extern "C" __global__ void __miss__ambientocclusion_shadow() {
	TraceInfoAmbientOcclusion* trace_info = PackedPointer<TraceInfoAmbientOcclusion>::from_payloads01();

	trace_info->visibility = 1.0f;
}


extern "C" __global__ void __anyhit__ambientocclusion_normal() {
	TraceInfoBasic const* trace_info = PackedPointer<TraceInfoBasic>::from_payloads01();

	ShadingOperation shade_op(trace_info->rng);

	generic_forward0_anyhit<TraceInfoBasic>(shade_op,trace_info);
}
extern "C" __global__ void __anyhit__ambientocclusion_shadow() {
	TraceInfoAmbientOcclusion* trace_info = PackedPointer<TraceInfoAmbientOcclusion>::from_payloads01();

	ShadingOperation shade_op(trace_info->rng);

	if (shade_op.stochastic_is_opaque()) {
		trace_info->visibility = 0.0f;
		optixTerminateRay();
	} else {
		optixIgnoreIntersection();
	}
}


extern "C" __global__ void __closesthit__ambientocclusion_normal() {
	TraceInfoBasic* trace_info = PackedPointer<TraceInfoBasic>::from_payloads01();

	ShadingOperation shade_op(trace_info->rng);
	shade_op.compute_shade_info_pos_normals();

	generic_forward0_closesthit(shade_op,trace_info);

	Ray ray_shad = {
		shade_op.shade_info.pos_wld,
		trace_info->rng->get_coshemi(shade_op.shade_info.Nshad_wld)
	};
	offset_ray_orig( &ray_shad, shade_op.shade_info.Ngeom_wld );

	TraceInfoAmbientOcclusion trace_info_shad(trace_info->rng);
	PackedPointer<TraceInfoAmbientOcclusion> ptr = &trace_info_shad;
	optixTrace(
		interface.traversable,

		to_float3(ray_shad.orig), to_float3(ray_shad.dir),

		0.0f, 4.0f,//std::numeric_limits<float>::infinity(),
		0.0f,

		OptixVisibilityMask(0b11111111),

		OptixRayFlags::OPTIX_RAY_FLAG_NONE,
		1u, unsigned int(SUMMER_MAX_RAYTYPES),
		1u,

		ptr[0], ptr[1]
	);

	Vec4f color = Vec4f(Vec3f(trace_info_shad.visibility),1.0f);
	semiAtomicAdd(interface.camera.framebuffer.layers.lighting_integration+trace_info->index.pixel_flat,color);
}

  
}
