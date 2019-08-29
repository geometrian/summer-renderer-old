#include "helpers.hpp"


namespace Summer {


/*enum RAY_TYPE {
	PRIMARY = 0,
	SHADOW  = 1
};*/


inline static __device__ Vec4f sample_texture(cudaTextureObject_t texture, Vec2f const& texc) {
	float4 tap = tex2D<float4>( texture, texc.x,texc.y );
	return Vec4f(tap.x,tap.y,tap.z,tap.w);
}


inline static __device__ Mat3x3f get_matr_TBN(Vec3f const verts[3], Vec2f const texcs[3], Vec3f const& Ngeom,Vec3f const& Nshad) {
	//Solve for T and B.
	Mat2x2f matr_uvs    = Mat2x2f( texcs[1]-texcs[0], texcs[2]-texcs[0] );
	Mat3x2f matr_deltas = Mat3x2f( verts[1]-verts[0], verts[2]-verts[0] );

	//	Check for the UVs to be well-formed, too.  TODO: maybe more elegant somehow?
	float divisor = matr_uvs[0][0]*matr_uvs[1][1] - matr_uvs[0][1]*matr_uvs[1][0];
	if (divisor==0) divisor=0.0001f;
	Mat2x2f matr_uvs_inv = Mat2x2f(
		 matr_uvs[1][1], -matr_uvs[0][1],
		-matr_uvs[1][0],  matr_uvs[0][0]
	) / divisor;
	//	Solve the system.
	Mat3x2f matr_TB = matr_deltas * matr_uvs_inv;

	//Ensure the handedness of the frame is correct.
	Vec3f B_from_cross = glm::cross(Ngeom,matr_TB[0]);
	if (glm::dot(B_from_cross,matr_TB[1])>=0.0f);
	else matr_TB[1]=-matr_TB[1];

	//Make the frame.
	Mat3x3f matr_TBN = Mat3x3f( matr_TB[0], matr_TB[1], Nshad );

	//Partially orthogonalize it to ensure everything is at-least perpendicular to the normals.
	matr_TBN[2] = glm::normalize( matr_TBN[2]                                                 );
	matr_TBN[0] = glm::normalize( matr_TBN[0] - glm::dot(matr_TBN[0],matr_TBN[2])*matr_TBN[0] );
	matr_TBN[1] = glm::normalize( matr_TBN[1] - glm::dot(matr_TBN[1],matr_TBN[2])*matr_TBN[1] );

	//Done.
	return matr_TBN;
}


static __device__ Scene::ShadePoint get_shade_info(DataSBT_HitOps const& data) {
	Vec3u indices;
	{
		unsigned int prim_index = optixGetPrimitiveIndex();
		unsigned int tmp = 3u * prim_index;
		if        (data.buffers_descriptor.type_indices == 0b01u) {
			//16-bit indices, most-common case
			indices = Vec3u(
				data.indices.u16[tmp   ],
				data.indices.u16[tmp+1u],
				data.indices.u16[tmp+2u]
			);
		} else if (data.buffers_descriptor.type_indices == 0b10u) {
			//32-bit indices
			indices = Vec3u(
				data.indices.u32[tmp   ],
				data.indices.u32[tmp+1u],
				data.indices.u32[tmp+2u]
			);
		} else {
			//No indices
			indices = Vec3u( tmp, tmp+1u, tmp+2u );
		}
	}

	Vec3f bary;
	{
		float2 bary_st = optixGetTriangleBarycentrics();
		bary = Vec3f( 1.0f-bary_st.x-bary_st.y, bary_st.x, bary_st.y);
	}

	bool has_texc0s = true;
	Vec2f texc0s[3];
	Vec2f texc0;
	{
		if        (data.buffers_descriptor.type_texcoords0 == 0b11u) {
			//Floating-point, most-common case
			texc0s[0] = data.texcoords0.f32x2[indices.x];
			texc0s[1] = data.texcoords0.f32x2[indices.y];
			texc0s[2] = data.texcoords0.f32x2[indices.z];
			texc0 = bary.x*texc0s[0] + bary.y*texc0s[1] + bary.z*texc0s[2];
		} else if (data.buffers_descriptor.type_texcoords0 == 0b00u) {
			//None
			texc0 = Vec2f(0.0f,0.0f);
			has_texc0s = false;
		} else if (data.buffers_descriptor.type_texcoords0 == 0b00u) {
			//16-bit indices
			texc0s[0] = Vec2f(data.texcoords0.u16x2[indices.x]);
			texc0s[1] = Vec2f(data.texcoords0.u16x2[indices.y]);
			texc0s[2] = Vec2f(data.texcoords0.u16x2[indices.z]);
			texc0 = ( bary.x*texc0s[0] + bary.y*texc0s[1] + bary.z*texc0s[2] ) * (1.0f/65535.0f);
		} else {
			//8-bit indices
			texc0s[0] = Vec2f(data.texcoords0.u8x2[indices.x]);
			texc0s[1] = Vec2f(data.texcoords0.u8x2[indices.y]);
			texc0s[2] = Vec2f(data.texcoords0.u8x2[indices.z]);
			texc0 = ( bary.x*texc0s[0] + bary.y*texc0s[1] + bary.z*texc0s[2] ) * (1.0f/255.0f);
		}
	}

	Scene::MaterialBase::InterfaceGPU const* material;
	{
		Scene::MaterialBase::InterfaceGPU const* materials = reinterpret_cast<Scene::MaterialBase::InterfaceGPU const*>(interface.materials);
		material = materials + data.material_index;
	}

	Vec3f verts[3];
	{
		verts[0] = from_float3(optixTransformPointFromObjectToWorldSpace(to_float3( data.verts[indices.x] )));
		verts[1] = from_float3(optixTransformPointFromObjectToWorldSpace(to_float3( data.verts[indices.y] )));
		verts[2] = from_float3(optixTransformPointFromObjectToWorldSpace(to_float3( data.verts[indices.z] )));
	}

	Vec3f Ngeom;
	{
		Vec3f u = verts[1] - verts[0];
		Vec3f v = verts[2] - verts[0];
		Ngeom = glm::normalize(glm::cross(u,v));
	}

	Vec3f Nshad;
	{
		Vec3f Nshad0_obj = data.norms[indices.x];
		Vec3f Nshad1_obj = data.norms[indices.y];
		Vec3f Nshad2_obj = data.norms[indices.z];
		Vec3f Nshad_obj = bary.x*Nshad0_obj + bary.y*Nshad1_obj + bary.z*Nshad2_obj;
		Nshad = from_float3(optixTransformNormalFromObjectToWorldSpace(to_float3(Nshad_obj)));

		if (has_texc0s) {
			switch (material->type) {
				case Scene::MaterialBase::TYPE::METALLIC_ROUGHNESS_RGBA: {
					if (material->metallic_roughness_rgba.normalmap!=0) {
						Mat3x3f matr_TBN = get_matr_TBN( verts, texc0s, Ngeom,Nshad );

						Vec3f normal_tangspace = sample_texture( material->metallic_roughness_rgba.normalmap, texc0 );
						normal_tangspace = normal_tangspace*2.0f - Vec3f(1.0f);

						Nshad = matr_TBN * normal_tangspace;
					}
					break;
				}
			}
		}

		Nshad = glm::normalize(Nshad);
	}

	return { indices, bary, texc0, material, Ngeom,Nshad };
}


inline static __device__ void write_rgba(Vec4f const& rgba) {
	uint32_t index = optixGetPayload_0();
	interface.camera.framebuffer.rgba.ptr[index] = pack_sRGB_A(rgba);
}


__device__ Vec4f Scene::MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::get_albedo(ShadePoint const* hit) const {
	Vec4f base_color = base_color_factor;
	if (base_color_texture!=0) base_color*=sample_texture( base_color_texture, hit->texc0 );
	return base_color;
}
__device__ Vec3f Scene::MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::emission(ShadePoint const* hit) const {
	Vec3f emission = emission_factor;
	if (emission_texture!=0) emission*=Vec3f(sample_texture( emission_texture, hit->texc0 ));
	return emission;
}
__device__ Vec4f Scene::MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::evaluate(ShadePointEvaluate const* hit) const {
	//https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-b-brdf-implementation

	//Load basic BRDF parameters
	Vec4f base_color;
	float metallic, roughness;
	{
		base_color = base_color_factor;
		if (base_color_texture!=0) base_color*=sample_texture( base_color_texture, hit->point.texc0 );

		metallic = metallic_factor;
		if (metallic_texture!=0) metallic*=sample_texture( metallic_texture, hit->point.texc0 ).x;

		roughness = roughness_factor;
		if (roughness_texture!=0) roughness*=sample_texture( roughness_texture, hit->point.texc0 ).y;
	}

	//Calculate diffuse and specular parameters
	Vec3f c_diff, F_0;
	{
		Vec3f spec_dielectric = Vec3f(0.04f);
		Vec3f black           = Vec3f(0.00f);

		c_diff = glm::mix( Vec3f(base_color)*(1.0f-spec_dielectric.r),black,             metallic );
		F_0    = glm::mix( spec_dielectric,                           Vec3f(base_color), metallic );
	}

	//Calculate BSDF intermediates
	Vec3f H;
	float alpha, alpha_sq;
	{
		H = glm::normalize( hit->w_i + hit->w_o );

		alpha = roughness * roughness;
		alpha_sq = alpha*alpha;
	}

	//Calculate Fresnel
	Vec3f F = F_0 + (Vec3f(1.0f)-F_0)*std::powf( 1.0f-glm::dot(hit->w_o,H), 5.0f );

	//Calculate diffuse component (Lambertian)
	Vec3f f_diffuse;
	{
		Vec3f diffuse = c_diff * RECIP_PI;
		f_diffuse = (Vec3f(1.0f)-F) * diffuse;
	}

	//Calculate specular component (microfacet)
	Vec3f f_specular;
	{
		float NdotL = glm::clamp(glm::dot(hit->point.Nshad,hit->w_i),0.0f,1.0f);
		float NdotV = glm::clamp(glm::dot(hit->point.Nshad,hit->w_o),0.0f,1.0f);
		float NdotH = glm::clamp(glm::dot(hit->point.Nshad,H       ),0.0f,1.0f);
		float LdotH = glm::clamp(glm::dot(hit->w_i,        H       ),0.0f,1.0f);

		//D (Trowbridge–Reitz)
		float denom = square(NdotH)*(alpha_sq-1.0f) + 1.0f;
		float D = alpha_sq / (PI*denom*denom);

		//V
		float k = alpha * 0.5f;
		auto G1 = [](float dotNX, float k) -> float {
			return 1.0f / (dotNX*(1.0f-k)+k);
		};
		float vis = G1(NdotL,k) * G1(NdotV,k);

		f_specular = F * D * vis;
	}

	Vec4f f = Vec4f( f_diffuse + f_specular, base_color.a );
	return f;
}
//__device__ Vec4f Scene::MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::interact(ShadePointInteract*       hit) const {}


__device__ Vec4f Scene::MaterialBase::InterfaceGPU::get_albedo(ShadePoint const* hit) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.get_albedo(hit);
		default:
			return Vec4f(1,0,1,1);
	}
}
__device__ Vec3f Scene::MaterialBase::InterfaceGPU::emission(ShadePoint const* hit) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.emission(hit);
		default:
			return Vec4f(1,0,1,1);
	}
}
__device__ Vec4f Scene::MaterialBase::InterfaceGPU::evaluate(ShadePointEvaluate const* hit) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.evaluate(hit);
		default:
			return Vec4f(1,0,1,1);
	}
}


extern "C" __global__ void __raygen__primary() {
	uint3 thread_index = optixGetLaunchIndex();

	uint32_t index = thread_index.y*interface.camera.framebuffer.res[0] + thread_index.x;

	Vec2f res = Vec2f(interface.camera.framebuffer.res);
	Vec2f uv = Vec2f(thread_index.x+0.5f,thread_index.y+0.5f) / res;
	float aspect = res.x / res.y;

	Vec3f const& pos = interface.camera.lookat.position;
	Vec3f const& cen = interface.camera.lookat.center;
	Vec3f        up  = interface.camera.lookat.up;

	Vec3f dir = glm::normalize(cen-pos);
	Vec3f x   = glm::normalize(glm::cross(dir,up));
	up = glm::cross(x,dir);

	Vec3f out = glm::normalize(
		dir + ((uv.x-0.5f)*aspect)*x + (uv.y-0.5f)*up
	);

	optixTrace(
		interface.traversable,

		to_float3(pos), to_float3(out),

		0.0f, std::numeric_limits<float>::infinity(),
		0.0f,

		OptixVisibilityMask(0b11111111),

		OptixRayFlags::OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		0u, 0u,
		0u,

		index
	);

	//interface.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( Vec3f(0.5f), 1.0f ));
}


extern "C" __global__ void __miss__primary() {
	uint32_t index = optixGetPayload_0();
	interface.camera.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( Vec3f(0.5f), 1.0f ));
}

extern "C" __global__ void __miss__shadow () {
	uint32_t index = optixGetPayload_0();
	interface.camera.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( Vec3f(0.5f), 1.0f ));
}


extern "C" __global__ void __closesthit__triangle_barycentric() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);
	write_rgba(Vec4f( shade_point.bary, 1.0f ));
}
extern "C" __global__ void __closesthit__texc0() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);
	write_rgba(Vec4f( shade_point.texc0,0.0f, 1.0f ));
}
extern "C" __global__ void __closesthit__Ngeom() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);
	write_rgba(Vec4f( shade_point.Ngeom, 1.0f ));
}
extern "C" __global__ void __closesthit__Nshad() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);
	write_rgba(Vec4f( shade_point.Nshad, 1.0f ));
}
extern "C" __global__ void __closesthit__albedo() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);
	Vec4f albedo = shade_point.material->get_albedo(&shade_point);
	write_rgba(albedo);
}
extern "C" __global__ void __closesthit__radiance() {
	DataSBT_HitOps const& data = *reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());
	Scene::ShadePoint shade_point = get_shade_info(data);

	//write_rgba(Vec4f( shade_point.bary, 1.0f ));
	//write_rgba(Vec4f( shade_point.texc0,0.0f, 1.0f ));
	//write_rgba(Vec4f( Vec3f(shade_point.texc0.y), 1.0f ));
	//write_rgba(Vec4f( shade_point.Ngeom, 1.0f ));
	//write_rgba(Vec4f( shade_point.Nshad, 1.0f ));

	//write_rgba(Vec4f( Vec3f(data.sbtentry_index/20.0f), 1.0f ));

	//unsigned int prim_index = optixGetPrimitiveIndex();
	//write_rgba(Vec4f( Vec3f(prim_index/50000.0f), 1.0f ));

	/*Vec3u indices = calc_indices(data);
	if (indices.x==0) {
		write_rgba(Vec4f( 1,0,1, 1.0f ));
	} else {
		write_rgba(Vec4f( Vec3f(indices)*0.4f, 1.0f ));
	}*/

	/*switch (data.material_index) {
		case 0:  write_rgba(Vec4f(1,0,0,1)); break;
		case 1:  write_rgba(Vec4f(0,1,0,1)); break;
		default: write_rgba(Vec4f(1,0,1,1)); break;
	}*/

	#if 0
		Vec4f albedo = shade_point.material->get_albedo(&shade_point);
		albedo.a = 1.0f;
		write_rgba(albedo);
	#endif
	#if 1
		Vec3f L = glm::normalize(Vec3f(1,2,1));
		float3 Vtmp = optixGetWorldRayDirection();
		Vec3f V = -Vec3f(Vtmp.x,Vtmp.y,Vtmp.z);

		Scene::ShadePointEvaluate hit = { shade_point, L,V };
		Vec4f bsdf = shade_point.material->evaluate(&hit);
		bsdf.a = 1.0f;

		#if 1
			Vec3f Li = Vec3f(20.0f);

			Vec3f Lo = Li * Vec3f(bsdf) * glm::dot(L,shade_point.Nshad);
			Lo += shade_point.material->emission(&shade_point);

			write_rgba(Vec4f(Lo,1.0f));
		#else
			write_rgba(bsdf);
		#endif
	#endif
}

extern "C" __global__ void __anyhit__radiance() {
	uint32_t index = optixGetPayload_0();
	interface.camera.framebuffer.rgba.ptr[index] = pack_sRGB_A(Vec4f( 0.5f,0.0f,0.0f, 1.0f ));
}

  
}
