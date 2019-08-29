#pragma once


#include "helpers.cuh"


namespace Summer {


/*extern "C" __global__ void __closesthit__Ngeom() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);
	write_rgba(Vec4f( shade_point.Ngeom, 1.0f ));
}
extern "C" __global__ void __closesthit__Nshad() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);
	write_rgba(Vec4f( shade_point.Nshad, 1.0f ));
}*/

extern "C" __global__ void __closesthit__normals() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	write_rgba(Vec4f( shade_point.Nshad, 1.0f ));
}


}
