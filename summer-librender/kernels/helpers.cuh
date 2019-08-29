#pragma once


#include "stdafx.cuh"


namespace Summer {


inline static __device__ float3 to_float3  (Vec3f  const& vec) { return { vec.x, vec.y, vec.z }; }
inline static __device__ Vec3f  from_float3(float3 const& vec) { return { vec.x, vec.y, vec.z }; }


template<typename Tout, typename Tin>
inline static __device__ Tout bit_cast(Tin const& value) {
	static_assert(sizeof(Tout)==sizeof(Tin),"Implementation error!");
	Tout result; memcpy(&result,&value,sizeof(Tin));
	return result;
}


#if 0
// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };
  
static __forceinline__ __device__ void *unpackPointer( uint32_t i0, uint32_t i1 ) {
	const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
	void*           ptr = reinterpret_cast<void*>( uptr ); 
	return ptr;
}
static __forceinline__ __device__ void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 ) {
	const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

template<typename T> static __forceinline__ __device__ T *getPRD() { 
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}
#endif

inline static __device__ uint32_t pack_sRGB_A(Vec4f const& srgb_a) {
	Vec4u discrete = Vec4u(glm::clamp( Vec4i(srgb_a * 255.0f), Vec4i(0),Vec4i(255) ));
	return (discrete.a<<24) | (discrete.b<<16) | (discrete.g<<8) | discrete.r;
}


template<typename T>
inline static __device__ T square(T const& value) { return value*value; }


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


}