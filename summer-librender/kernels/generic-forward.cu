#pragma once


#include "helpers.cuh"
#include "shading.cuh"


namespace Summer {


class TraceInfoBasic { //Note not `final`
	public:
		struct {
			union {
				Vec2u pixel_2D;
				Vec3u thread_3D;
			};
			uint32_t thread_flat;
			uint32_t pixel_flat;
		} index;

		RNG* rng;

	public:
		__device__ TraceInfoBasic() {
			{
				uint3 tmp = optixGetLaunchIndex();
				index.thread_3D   = Vec3u(tmp.x,tmp.y,tmp.z);
				index.pixel_flat  = index.thread_3D.y*interface.camera.framebuffer.res[0] + index.thread_3D.x;
				index.thread_flat = interface.camera.framebuffer.res[1]*interface.camera.framebuffer.res[0]*index.thread_3D.z + index.pixel_flat;
			}

			rng = interface.camera.framebuffer.layers.rngs + index.thread_flat;
		}
};


template<class TraceInfo>
__device__ void generic_forward0_raygen(TraceInfo& trace_info) {
	Vec2f pixel = Vec2f(trace_info.index.pixel_2D);
	pixel += Vec2f( trace_info.rng->get_uniform(), trace_info.rng->get_uniform() );
	Ray ray = interface.camera.get_ray(pixel);

	semiAtomicAdd(interface.camera.framebuffer.layers.sampling_weights_and_count+trace_info.index.pixel_flat,Vec2f(1.0f,1.0f));

	PackedPointer<TraceInfo> ptr = &trace_info;
	optixTrace(
		interface.traversable,

		to_float3(ray.orig), to_float3(ray.dir),

		0.0f, INFf,
		0.0f,

		OptixVisibilityMask(0b11111111),

		OptixRayFlags::OPTIX_RAY_FLAG_NONE,
		0u, unsigned int(SUMMER_MAX_RAYTYPES),
		0u,

		ptr[0], ptr[1]
	);
}


template<class TraceInfo>
__device__ TraceInfo* generic_forward0_miss() {
	TraceInfo* trace_info = PackedPointer<TraceInfo>::from_payloads01();

	auto& layers = interface.camera.framebuffer.layers;
	//                                         semiAtomicAdd(layers.sampling_weights_and_count+trace_info->index.pixel_flat,Vec2f(1.0f,1.0f));
	if (layers.scene_distance     !=nullptr) atomicAdd    (layers.scene_distance            +trace_info->index.pixel_flat,INFf);
	if (layers.scene_depth        !=nullptr) atomicAdd    (layers.scene_depth               +trace_info->index.pixel_flat,1.0f);
	//if (layers.stats_accelstruct  !=nullptr) semiAtomicAdd(layers.stats_accelstruct         +trace_info->index.pixel_flat,);
	//if (layers.stats_photondensity!=nullptr) atomicAdd    (layers.stats_photondensity       +trace_info->index.pixel_flat,);

	return trace_info;
}


template<class TraceInfo>
__device__ void generic_forward0_anyhit(ShadingOperation const& shade_op, TraceInfo const* trace_info) {
	auto& layers = interface.camera.framebuffer.layers;
	//if (layers.stats_accelstruct    !=nullptr) semiAtomicAdd(layers.stats_accelstruct    +trace_info->index.pixel_flat,);
	if (layers.stats_depthcomplexity!=nullptr) atomicAdd    (layers.stats_depthcomplexity+trace_info->index.pixel_flat,1.0f);

	if (shade_op.stochastic_is_opaque()); else optixIgnoreIntersection();
}


//Requires `shade_op.shade_info.compute_normals()`.
template<class TraceInfo>
__device__ void generic_forward0_closesthit(ShadingOperation const& shade_op, TraceInfo const* trace_info) {
	auto& layers = interface.camera.framebuffer.layers;
	//                                                            semiAtomicAdd(layers.sampling_weights_and_count            +trace_info->index.pixel_flat,Vec2f(1.0f,1.0f));
	//if (layers.scene_distance                        !=nullptr) atomicAdd    (layers.scene_distance                        +trace_info->index.pixel_flat,);
	//if (layers.scene_depth                           !=nullptr) atomicAdd    (layers.scene_depth                           +trace_info->index.pixel_flat,);
	if (layers.scene_normals_geometric               !=nullptr) semiAtomicAdd(layers.scene_normals_geometric               +trace_info->index.pixel_flat,shade_op.shade_info.Ngeom_wld);
	if (layers.scene_normals_shading                 !=nullptr) semiAtomicAdd(layers.scene_normals_shading                 +trace_info->index.pixel_flat,shade_op.shade_info.Nshad_wld);
	//if (layers.scene_motion_vectors                  !=nullptr) semiAtomicAdd(layers.scene_motion_vectors                  +trace_info->index.pixel_flat,);
	if (layers.scene_triangle_barycentric_coordinates!=nullptr) semiAtomicAdd(layers.scene_triangle_barycentric_coordinates+trace_info->index.pixel_flat,shade_op.shade_info.bary_2D);
	if (layers.scene_texture_coordinates             !=nullptr) semiAtomicAdd(layers.scene_texture_coordinates             +trace_info->index.pixel_flat,shade_op.shade_info.texc0);
	if (layers.scene_albedo                          !=nullptr) semiAtomicAdd(layers.scene_albedo                          +trace_info->index.pixel_flat,shade_op.albedo);
	//if (layers.scene_fresnel_term                    !=nullptr) {
	//	fresnel = ;
	//	semiAtomicAdd(layers.scene_fresnel_term+trace_info->index.pixel_flat,fresnel);
	//}
	//if (layers.stats_accelstruct                     !=nullptr) semiAtomicAdd(layers.stats_accelstruct                     +trace_info->index.pixel_flat,);
	//if (layers.stats_depthcomplexity                 !=nullptr) atomicAdd    (layers.stats_depthcomplexity                 +trace_info->index.pixel_flat,);
	//if (layers.stats_photondensity                   !=nullptr) atomicAdd    (layers.stats_photondensity                   +trace_info->index.pixel_flat,);
}


}
