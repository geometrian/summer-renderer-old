#pragma once


#include "../rng.hpp"

#include "helpers.cuh"


namespace Summer {


__device__ void generic_forward0_raygen() {
	uint3 thread_index = optixGetLaunchIndex();
	uint32_t index = thread_index.y*interface.camera.framebuffer.res[0] + thread_index.x;

	RNG& rng = interface.camera.framebuffer.layers.rngs[index];

	Vec2f pixel = Vec2f( thread_index.x, thread_index.y );
	pixel += Vec2f( rng.get_next(), rng.get_next() );
	Ray ray = interface.camera.get_ray(pixel);

	semiAtomicAdd(interface.camera.framebuffer.layers.sampling_weights_and_count+index,Vec2f(1.0f,1.0f));

	optixTrace(
		interface.traversable,

		to_float3(ray.orig), to_float3(ray.dir),

		0.0f, INFf,
		0.0f,

		OptixVisibilityMask(0b11111111),

		OptixRayFlags::OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0u, unsigned int(SUMMER_MAX_RAYTYPES),
		0u,

		index
	);
}


__device__ void generic_forward0_miss() {
	uint32_t index = optixGetPayload_0();

	auto& layers = interface.camera.framebuffer.layers;
	//                                         semiAtomicAdd(layers.sampling_weights_and_count+index,Vec2f(1.0f,1.0f));
	if (layers.scene_distance     !=nullptr) atomicAdd    (layers.scene_distance            +index,INFf);
	if (layers.scene_depth        !=nullptr) atomicAdd    (layers.scene_depth               +index,1.0f);
	//if (layers.stats_accelstruct  !=nullptr) semiAtomicAdd(layers.stats_accelstruct         +index,);
	//if (layers.stats_photondensity!=nullptr) atomicAdd    (layers.stats_photondensity       +index,);
}


__device__ void generic_forward0_anyhit() {
	uint32_t index = optixGetPayload_0();

	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	auto& layers = interface.camera.framebuffer.layers;
	//if (layers.stats_accelstruct    !=nullptr) semiAtomicAdd(layers.stats_accelstruct    +index,);
	if (layers.stats_depthcomplexity!=nullptr) atomicAdd    (layers.stats_depthcomplexity+index,1.0f);
}


__device__ void generic_forward0_closesthit() {
	uint32_t index = optixGetPayload_0();

	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	auto& layers = interface.camera.framebuffer.layers;
	//                                                            semiAtomicAdd(layers.sampling_weights_and_count            +index,Vec2f(1.0f,1.0f));
	//if (layers.scene_distance                        !=nullptr) atomicAdd    (layers.scene_distance                        +index,);
	//if (layers.scene_depth                           !=nullptr) atomicAdd    (layers.scene_depth                           +index,);
	if (layers.scene_normals_geometric               !=nullptr) semiAtomicAdd(layers.scene_normals_geometric               +index,shade_point.Ngeom);
	if (layers.scene_normals_shading                 !=nullptr) semiAtomicAdd(layers.scene_normals_shading                 +index,shade_point.Nshad);
	//if (layers.scene_motion_vectors                  !=nullptr) semiAtomicAdd(layers.scene_motion_vectors                  +index,);
	if (layers.scene_triangle_barycentric_coordinates!=nullptr) semiAtomicAdd(layers.scene_triangle_barycentric_coordinates+index,shade_point.bary_2D);
	if (layers.scene_texture_coordinates             !=nullptr) semiAtomicAdd(layers.scene_texture_coordinates             +index,shade_point.texc0);
	if (layers.scene_albedo                          !=nullptr) {
		Vec4f albedo = shade_point.material->get_albedo(&shade_point);
		semiAtomicAdd(layers.scene_albedo+index,albedo);
	}
	//if (layers.scene_fresnel_term                    !=nullptr) {
	//	fresnel = ;
	//	semiAtomicAdd(layers.scene_fresnel_term+index,fresnel);
	//}
	//if (layers.stats_accelstruct                     !=nullptr) semiAtomicAdd(layers.stats_accelstruct                     +index,);
	//if (layers.stats_depthcomplexity                 !=nullptr) atomicAdd    (layers.stats_depthcomplexity                 +index,);
	//if (layers.stats_photondensity                   !=nullptr) atomicAdd    (layers.stats_photondensity                   +index,);
}


}
