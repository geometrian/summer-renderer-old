#pragma once


#include "helpers.cuh"


namespace Summer {


extern "C" __global__ void __closesthit__tri_bary() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	write_rgba(Vec4f( shade_point.bary, 1.0f ));
}


}
