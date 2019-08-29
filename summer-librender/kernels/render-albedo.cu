#pragma once


#include "helpers.cuh"


namespace Summer {


extern "C" __global__ void __closesthit__albedo() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	Vec4f albedo = shade_point.material->get_albedo(&shade_point);
	albedo = Vec4f( Vec3f(albedo)*albedo.a, 1.0f );

	write_rgba(albedo);
}


}
