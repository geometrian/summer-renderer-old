#pragma once


#include "generic-forward.cu"


namespace Summer {


class TraceInfoPathtraceNormal final : public TraceInfoBasic {
	public:
		uint32_t ray_depth;

		Vec3f throughput;
		float pdf;

		Vec3f Lo;

	public:
		__device__ TraceInfoPathtraceNormal() :
			TraceInfoBasic(),
			ray_depth(0u), throughput(1.0f),pdf(1.0f), Lo(Vec3f(0.0f))
		{}
};
class TraceInfoPathtraceShadow final {
	public:
		RNG*const rng;

		float visibility;

	public:
		__device__ explicit TraceInfoPathtraceShadow(RNG* rng) : rng(rng) {}
};


extern "C" __global__ void __raygen__pathtracing() {
	TraceInfoPathtraceNormal trace_info;
	generic_forward0_raygen<TraceInfoPathtraceNormal>(trace_info);

	semiAtomicAdd(interface.camera.framebuffer.layers.lighting_integration+trace_info.index.pixel_flat,Vec4f(trace_info.Lo,1.0f));
}


extern "C" __global__ void __miss__pathtracing_normal() {
	TraceInfoPathtraceNormal* trace_info = generic_forward0_miss<TraceInfoPathtraceNormal>();

	#if 0
		Vec3f Li = Vec3f(0.1f);
		if (trace_info->ray_depth>0u);
		else {
			trace_info->Lo += Li;
		}
	#else
		Vec3f Li = 10.0f*Vec3f(5.0f,6.0f,8.0f);
		trace_info->Lo += Li * trace_info->throughput / trace_info->pdf;
	#endif
}
extern "C" __global__ void __miss__pathtracing_shadow() {
	TraceInfoPathtraceShadow* trace_info = PackedPointer<TraceInfoPathtraceShadow>::from_payloads01();

	trace_info->visibility = 1.0f;
}


extern "C" __global__ void __anyhit__pathtracing_normal() {
	TraceInfoPathtraceNormal const* trace_info = PackedPointer<TraceInfoPathtraceNormal>::from_payloads01();

	ShadingOperation shade_op(trace_info->rng);

	generic_forward0_anyhit<TraceInfoPathtraceNormal>(shade_op,trace_info);
}
extern "C" __global__ void __anyhit__pathtracing_shadow() {
	TraceInfoPathtraceShadow* trace_info = PackedPointer<TraceInfoPathtraceShadow>::from_payloads01();

	ShadingOperation shade_op(trace_info->rng);

	if (shade_op.stochastic_is_opaque()) {
		trace_info->visibility = 0.0f;
		optixTerminateRay();
	} else {
		optixIgnoreIntersection();
	}
}


extern "C" __global__ void __closesthit__pathtracing_normal() {
	TraceInfoPathtraceNormal* trace_info = PackedPointer<TraceInfoPathtraceNormal>::from_payloads01();

	ShadingOperation shade_op(trace_info->rng);
	shade_op.compute_shade_info_pos_normals();

	if (trace_info->ray_depth>0u); else {
		generic_forward0_closesthit(shade_op,trace_info);
	}

	float3 Vtmp = optixGetWorldRayDirection();
	Vec3f V = glm::normalize(-Vec3f(Vtmp.x,Vtmp.y,Vtmp.z));

	shade_op.w_o = V;
	shade_op.fix_normals_from_w_o();

	//Emission
	trace_info->Lo += shade_op.compute_edf_emission() * trace_info->throughput;

	//Direct lighting
	#if 1
	{
		Vec3f light_pos = Vec3f(1000,2000,1000);
		//Vec3f light_pos = Vec3f(200,1000,-100);
		//Vec3f light_pos = Vec3f(0,2000,0);
		Vec3f L = glm::normalize( light_pos - shade_op.shade_info.pos_wld );

		TraceInfoPathtraceShadow info_shad(trace_info->rng);

		Ray ray_shad = { shade_op.shade_info.pos_wld, L };
		offset_ray_orig( &ray_shad, shade_op.shade_info.Ngeom_wld );

		PackedPointer<TraceInfoPathtraceShadow> ptr = &info_shad;
		optixTrace(
			interface.traversable,

			to_float3(ray_shad.orig), to_float3(ray_shad.dir),

			0.0f, std::numeric_limits<float>::infinity(),
			0.0f,

			OptixVisibilityMask(0b11111111),

			OptixRayFlags::OPTIX_RAY_FLAG_NONE,
			1u, unsigned int(SUMMER_MAX_RAYTYPES),
			1u,

			ptr[0], ptr[1]
		);

		Vec3f Li = Vec3f(100.0f);

		shade_op.w_i = L;
		Vec4f bsdf = shade_op.compute_bsdf_evaluate();

		trace_info->Lo += Li * trace_info->throughput * Vec3f(bsdf) * glm::abs(glm::dot( shade_op.shade_info.Nshad_wld, L )) * info_shad.visibility / trace_info->pdf;
	}
	#endif

	#if 1
	if (trace_info->ray_depth<3u) {
		//Indirect lighting

		Ray ray_ind = { shade_op.shade_info.pos_wld, trace_info->rng->get_coshemi(shade_op.shade_info.Nshad_wld) };
		offset_ray_orig( &ray_ind, shade_op.shade_info.Ngeom_wld );

		shade_op.w_i = ray_ind.dir;
		Vec4f bsdf = shade_op.compute_bsdf_evaluate();
		trace_info->throughput *= Vec3f(bsdf) * glm::abs(glm::dot( shade_op.shade_info.Nshad_wld, shade_op.w_i ));

		PackedPointer<TraceInfoPathtraceNormal> ptr = trace_info;
		++trace_info->ray_depth;
		//trace_info->pdf *= RECIP_PI;
		optixTrace(
			interface.traversable,

			to_float3(ray_ind.orig), to_float3(ray_ind.dir),

			0.0f, std::numeric_limits<float>::infinity(),
			0.0f,

			OptixVisibilityMask(0b11111111),

			OptixRayFlags::OPTIX_RAY_FLAG_NONE,
			0u, unsigned int(SUMMER_MAX_RAYTYPES),
			0u,

			ptr[0], ptr[1]
		);
		//trace_info->pdf /= RECIP_PI;
		//--trace_info->ray_depth;
	}
	#endif

	//trace_info->Lo += trace_info->rng->get_coshemi(shade_op.shade_info.Nshad_wld);
}

  
}
