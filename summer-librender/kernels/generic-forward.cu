#pragma once


#include "../rng.hpp"

#include "helpers.cuh"


namespace Summer {


class TraceInfoBasic final {
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
};


__device__ void generic_forward0_raygen() {
	TraceInfoBasic info;

	{
		uint3 tmp = optixGetLaunchIndex();
		info.index.thread_3D   = Vec3u(tmp.x,tmp.y,tmp.z);
		info.index.pixel_flat  = info.index.thread_3D.y*interface.camera.framebuffer.res[0] + info.index.thread_3D.x;
		info.index.thread_flat = interface.camera.framebuffer.res[1]*interface.camera.framebuffer.res[0]*info.index.thread_3D.z + info.index.pixel_flat;
	}

	info.rng = interface.camera.framebuffer.layers.rngs + info.index.thread_flat;

	Vec2f pixel = Vec2f(info.index.pixel_2D);
	pixel += Vec2f( info.rng->get_next(), info.rng->get_next() );
	Ray ray = interface.camera.get_ray(pixel);

	semiAtomicAdd(interface.camera.framebuffer.layers.sampling_weights_and_count+info.index.pixel_flat,Vec2f(1.0f,1.0f));

	PackedPointer<TraceInfoBasic> ptr = &info;
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


__device__ TraceInfoBasic const* generic_forward0_miss() {
	TraceInfoBasic const* info = PackedPointer<TraceInfoBasic>::from_payloads01();

	auto& layers = interface.camera.framebuffer.layers;
	//                                         semiAtomicAdd(layers.sampling_weights_and_count+info->index.pixel_flat,Vec2f(1.0f,1.0f));
	if (layers.scene_distance     !=nullptr) atomicAdd    (layers.scene_distance            +info->index.pixel_flat,INFf);
	if (layers.scene_depth        !=nullptr) atomicAdd    (layers.scene_depth               +info->index.pixel_flat,1.0f);
	//if (layers.stats_accelstruct  !=nullptr) semiAtomicAdd(layers.stats_accelstruct         +info->index.pixel_flat,);
	//if (layers.stats_photondensity!=nullptr) atomicAdd    (layers.stats_photondensity       +info->index.pixel_flat,);

	return info;
}


__device__ TraceInfoBasic const* generic_forward0_anyhit() {
	TraceInfoBasic const* info = PackedPointer<TraceInfoBasic>::from_payloads01();

	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);
	Vec4f albedo = shade_point.material->get_albedo(&shade_point);
	if        (albedo.a==1.0f) {
	} else if (albedo.a!=0.0f) {
		if (info->rng->get_next() <= albedo.a) {
		} else {
			optixIgnoreIntersection();
		}
	} else {
		optixIgnoreIntersection();
	}

	auto& layers = interface.camera.framebuffer.layers;
	//if (layers.stats_accelstruct    !=nullptr) semiAtomicAdd(layers.stats_accelstruct    +info->index.pixel_flat,);
	if (layers.stats_depthcomplexity!=nullptr) atomicAdd    (layers.stats_depthcomplexity+info->index.pixel_flat,1.0f);

	return info;
}


__device__ TraceInfoBasic const* generic_forward0_closesthit() {
	TraceInfoBasic const* info = PackedPointer<TraceInfoBasic>::from_payloads01();

	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	auto& layers = interface.camera.framebuffer.layers;
	//                                                            semiAtomicAdd(layers.sampling_weights_and_count            +info->index.pixel_flat,Vec2f(1.0f,1.0f));
	//if (layers.scene_distance                        !=nullptr) atomicAdd    (layers.scene_distance                        +info->index.pixel_flat,);
	//if (layers.scene_depth                           !=nullptr) atomicAdd    (layers.scene_depth                           +info->index.pixel_flat,);
	if (layers.scene_normals_geometric               !=nullptr) semiAtomicAdd(layers.scene_normals_geometric               +info->index.pixel_flat,shade_point.Ngeom);
	if (layers.scene_normals_shading                 !=nullptr) semiAtomicAdd(layers.scene_normals_shading                 +info->index.pixel_flat,shade_point.Nshad);
	//if (layers.scene_motion_vectors                  !=nullptr) semiAtomicAdd(layers.scene_motion_vectors                  +info->index.pixel_flat,);
	if (layers.scene_triangle_barycentric_coordinates!=nullptr) semiAtomicAdd(layers.scene_triangle_barycentric_coordinates+info->index.pixel_flat,shade_point.bary_2D);
	if (layers.scene_texture_coordinates             !=nullptr) semiAtomicAdd(layers.scene_texture_coordinates             +info->index.pixel_flat,shade_point.texc0);
	if (layers.scene_albedo                          !=nullptr) {
		Vec4f albedo = shade_point.material->get_albedo(&shade_point);
		semiAtomicAdd(layers.scene_albedo+info->index.pixel_flat,albedo);
	}
	//if (layers.scene_fresnel_term                    !=nullptr) {
	//	fresnel = ;
	//	semiAtomicAdd(layers.scene_fresnel_term+info->index.pixel_flat,fresnel);
	//}
	//if (layers.stats_accelstruct                     !=nullptr) semiAtomicAdd(layers.stats_accelstruct                     +info->index.pixel_flat,);
	//if (layers.stats_depthcomplexity                 !=nullptr) atomicAdd    (layers.stats_depthcomplexity                 +info->index.pixel_flat,);
	//if (layers.stats_photondensity                   !=nullptr) atomicAdd    (layers.stats_photondensity                   +info->index.pixel_flat,);

	return info;
}


}
