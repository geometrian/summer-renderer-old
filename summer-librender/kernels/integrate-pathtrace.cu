#pragma once


#include "generic-forward.cu"


namespace Summer {


extern "C" __global__ void __raygen__pathtracing() {
	generic_forward0_raygen();
}


extern "C" __global__ void __miss__pathtracing_normal() {
	generic_forward0_miss();

	uint32_t index = optixGetPayload_0();
	semiAtomicAdd(interface.camera.framebuffer.layers.lighting_integration+index,Vec4f( Vec3f(0.1f), 1.0f ));
}
extern "C" __global__ void __miss__pathtracing_shadow() {
	float* visibility = PackedPointer<float>::from_payloads01();
	*visibility = 1.0f;
}


extern "C" __global__ void __anyhit__pathtracing_normal() {
	generic_forward0_anyhit();
}
extern "C" __global__ void __anyhit__pathtracing_shadow() {
	float* visibility = PackedPointer<float>::from_payloads01();
	*visibility = 0.01f;
}


extern "C" __global__ void __closesthit__pathtracing_normal() {
	generic_forward0_closesthit();

	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	//write_rgba(Vec4f( shade_point.bary, 1.0f ));
	//write_rgba(Vec4f( shade_point.texc0,0.0f, 1.0f ));
	//write_rgba(Vec4f( Vec3f(shade_point.texc0.y), 1.0f ));
	//write_rgba(Vec4f( shade_point.Ngeom, 1.0f ));
	//write_rgba(Vec4f( shade_point.Nshad, 1.0f ));

	//write_rgba(Vec4f( Vec3f(data.sbtentry_index/20.0f), 1.0f ));

	/*Vec3u indices = calc_indices(data);
	if (indices.x==0) {
		write_rgba(Vec4f( 1,0,1, 1.0f ));
	} else {
		write_rgba(Vec4f( Vec3f(indices)*0.4f, 1.0f ));
	}*/

	/*switch (data.buffers_descriptor.type_indices) {
		case 0b00u:
			write_rgba(Vec4f( 0.0f,0.0f,0.0f, 1.0f ));
			break;
		case 0b01u: //16-bit
			write_rgba(Vec4f( 0.0f,0.0f,1.0f, 1.0f ));
			break;
		case 0b10u: //32-bit
			write_rgba(Vec4f( 0.0f,1.0f,0.0f, 1.0f ));
			break;
		default: //error
			write_rgba(Vec4f( 1.0f,0.0f,1.0f, 1.0f ));
			break;
	}*/
	#if 0
		unsigned int prim_index = optixGetPrimitiveIndex();
		write_rgba(Vec4f( Vec3f(prim_index/50000.0f), 1.0f ));
	#endif
	#if 0
		write_rgba(Vec4f( Vec3f(shade_point.indices)/62663.0f, 1.0f ));
	#endif
	#if 0
	switch (data.material_index) {
		case 0:  write_rgba(Vec4f(1,0,0,1)); break;
		case 1:  write_rgba(Vec4f(0,1,0,1)); break;
		default: write_rgba(Vec4f(1,0,1,1)); break;
	}
	#endif
	#if 0
		Vec4f albedo = shade_point.material->get_albedo(&shade_point);
		albedo.a = 1.0f;
		write_rgba(albedo);
	#endif
	#if 1
		Vec3f light_pos = Vec3f(1000,2000,1000);

		Vec3f L = glm::normalize(light_pos-shade_point.pos);

		float visibility;
		#if 1
			PackedPointer<float> ptr = &visibility;

			uint32_t u0=ptr[0], u1=ptr[1];
			optixTrace(
				interface.traversable,

				to_float3(shade_point.pos+0.001f*L), to_float3(L),

				0.0f, std::numeric_limits<float>::infinity(),
				0.0f,

				OptixVisibilityMask(0b11111111),

				OptixRayFlags::OPTIX_RAY_FLAG_NONE,
				1u, 2u,
				1u,

				u0, u1//ptr[0], ptr[1]
			);
		#else
			visibility = 1.0f;
		#endif

		float3 Vtmp = optixGetWorldRayDirection();
		Vec3f V = -Vec3f(Vtmp.x,Vtmp.y,Vtmp.z);

		Scene::ShadePointEvaluate hit = { shade_point, L,V };
		Vec4f bsdf = shade_point.material->evaluate(&hit);

		bsdf *= visibility;

		bsdf.a = 1.0f;
		#if 1
			Vec3f Li = Vec3f(10.0f);

			Vec3f Lo = Li * Vec3f(bsdf) * glm::abs(glm::dot(L,shade_point.Nshad));
			Lo += shade_point.material->emission(&shade_point);

			Vec4f rgba = Vec4f(Lo,1.0f);
		#else
			Vec4f rgba = bsdf;
		#endif

		uint32_t index = optixGetPayload_0();
		semiAtomicAdd(interface.camera.framebuffer.layers.lighting_integration+index,rgba);
	#endif
}

  
}