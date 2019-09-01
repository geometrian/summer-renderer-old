#pragma once


#include "stdafx.cuh"

#include "../scene/materials/material.hpp"


namespace Summer {


inline static __device__ Vec4f sample_texture(cudaTextureObject_t texture, Vec2f const& texc) {
	float4 tap = tex2D<float4>( texture, texc.x,texc.y );
	return Vec4f( tap.x,tap.y,tap.z, tap.w );
}


inline static __device__ Mat3x3f _get_matr_TBN(Vec3f const verts[3], Vec2f const texcs[3], Vec3f const& Ngeom,Vec3f const& Nshad) {
	//Solve for T and B.
	Mat2x2f matr_uvs    = Mat2x2f( texcs[1]-texcs[0], texcs[2]-texcs[0] );
	Mat3x2f matr_deltas = Mat3x2f( verts[1]-verts[0], verts[2]-verts[0] );

	//	Check for the UVs to be well-formed, too.  TODO: maybe more elegant somehow?
	float divisor = matr_uvs[0][0]*matr_uvs[1][1] - matr_uvs[0][1]*matr_uvs[1][0];
	if (divisor==0.0f) divisor=0.0001f;
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
	//	Note: `matr_TBN[2]` is passed in and is assumed to be normalized already
	matr_TBN[0] = matr_TBN[0] - glm::dot(matr_TBN[0],matr_TBN[2])*matr_TBN[0];
	matr_TBN[1] = matr_TBN[1] - glm::dot(matr_TBN[1],matr_TBN[2])*matr_TBN[1];

	float lsq0 = glm::dot(matr_TBN[0],matr_TBN[0]);
	float lsq1 = glm::dot(matr_TBN[1],matr_TBN[1]);
	if ( lsq0>0.00001f && lsq1>0.00001f ) {
		matr_TBN[0] *= rsqrtf(lsq0);
		matr_TBN[1] *= rsqrtf(lsq1);
	} else {
		build_frame( matr_TBN[2], &(matr_TBN[0]), &(matr_TBN[1]) );
	}

	//Done.
	return matr_TBN;
}


class ShadingOperation;


class ShadeInfo final {
	friend class ShadingOperation;
	private:
		DataSBT_HitOps const* _data;
		Scene::MaterialBase::InterfaceGPU const* _material;

	public:
		//Note: this is `uint32_t`s because although indices are semantically `size_t`, OptiX does
		//	not support larger index types, and those can only make it slower.
		Vec3u indices_verts;
		uint32_t index_prim;

		union {
			Vec2f bary_2D;
			Vec3f bary_3D;
		};

		Vec3f verts_obj[3];
		Vec3f pos_wld;

		Vec2f texc0s[3];
		Vec2f texc0;
		bool has_texc0s;

		Vec3f Ngeom_wld;
		Vec3f Nshad_wld;

	public:
		__device__ ShadeInfo() {
			_data = reinterpret_cast<DataSBT_HitOps*>(optixGetSbtDataPointer());

			//TODO: compute lazily instead?
			Scene::MaterialBase::InterfaceGPU const* materials = reinterpret_cast<Scene::MaterialBase::InterfaceGPU const*>(interface.materials);
			_material = materials + _data->material_index;

			_compute_indices();
		}
		~ShadeInfo() = default;

	private:
		//Always computed.
		__device__ void _compute_indices() {
			index_prim = static_cast<uint32_t>(optixGetPrimitiveIndex());
			uint32_t tmp = 3u * index_prim;
			if        (_data->buffers_descriptor.type_indices == 0b01u) {
				//16-bit indices, most-common case
				indices_verts = Vec3u(
					_data->indices.u16[tmp   ],
					_data->indices.u16[tmp+1u],
					_data->indices.u16[tmp+2u]
				);
			} else if (_data->buffers_descriptor.type_indices == 0b10u) {
				//32-bit indices
				indices_verts = Vec3u(
					_data->indices.u32[tmp   ],
					_data->indices.u32[tmp+1u],
					_data->indices.u32[tmp+2u]
				);
			} else {
				//No indices
				indices_verts = Vec3u( tmp, tmp+1u, tmp+2u );
			}
		}
	public:
		//No extra dependencies.
		__device__ void compute_bary() {
			float2 bary_st = optixGetTriangleBarycentrics();
			bary_3D = Vec3f( 1.0f-bary_st.x-bary_st.y, bary_st.x, bary_st.y);
		}
		__device__ void compute_verts() {
			verts_obj[0] = _data->verts[indices_verts.x];
			verts_obj[1] = _data->verts[indices_verts.y];
			verts_obj[2] = _data->verts[indices_verts.z];
		}

		//Depends `.compute_bary()`.
		__device__ bool compute_texc0s() {
			if        (_data->buffers_descriptor.type_texcoords0 == 0b11u) {
				//Floating-point, most-common case
				texc0s[0] = _data->texcoords0.f32x2[indices_verts.x];
				texc0s[1] = _data->texcoords0.f32x2[indices_verts.y];
				texc0s[2] = _data->texcoords0.f32x2[indices_verts.z];
				texc0 = bary_3D.x*texc0s[0] + bary_3D.y*texc0s[1] + bary_3D.z*texc0s[2];
			} else if (_data->buffers_descriptor.type_texcoords0 == 0b00u) {
				//None
				texc0 = Vec2f(0.0f,0.0f);
				return has_texc0s=false;
			} else if (_data->buffers_descriptor.type_texcoords0 == 0b00u) {
				//16-bit indices
				texc0s[0] = Vec2f(_data->texcoords0.u16x2[indices_verts.x]);
				texc0s[1] = Vec2f(_data->texcoords0.u16x2[indices_verts.y]);
				texc0s[2] = Vec2f(_data->texcoords0.u16x2[indices_verts.z]);
				texc0 = ( bary_3D.x*texc0s[0] + bary_3D.y*texc0s[1] + bary_3D.z*texc0s[2] ) * (1.0f/65535.0f);
			} else {
				//8-bit indices
				texc0s[0] = Vec2f(_data->texcoords0.u8x2[indices_verts.x]);
				texc0s[1] = Vec2f(_data->texcoords0.u8x2[indices_verts.y]);
				texc0s[2] = Vec2f(_data->texcoords0.u8x2[indices_verts.z]);
				texc0 = ( bary_3D.x*texc0s[0] + bary_3D.y*texc0s[1] + bary_3D.z*texc0s[2] ) * (1.0f/255.0f);
			}
			return has_texc0s=true;
		}

		//Depends `.compute_bary()` and `.compute_verts()`.
		__device__ void compute_pos() {
			Vec3f pos_obj = bary_3D.x*verts_obj[0] + bary_3D.y*verts_obj[1] + bary_3D.z*verts_obj[2];
			pos_wld = from_float3(optixTransformPointFromObjectToWorldSpace(to_float3( pos_obj )));
		}

		//Depends `.compute_texc0()` and `.compute_verts()`.
		__device__ void compute_normals() {
			//Geometric normal
			Vec3f uf = verts_obj[1] - verts_obj[0];
			Vec3f vf = verts_obj[2] - verts_obj[0];
			Vec3f Ngeomf_obj = glm::cross(uf,vf);

			//	Normalize it
			#if 1
			float lsqf = glm::dot(Ngeomf_obj,Ngeomf_obj);
			if (lsqf>0.00001f) {
				Ngeomf_obj *= rsqrtf(lsqf);
			} else {
				Vec3d ud = Vec3d(verts_obj[1]) - Vec3d(verts_obj[0]);
				Vec3d vd = Vec3d(verts_obj[2]) - Vec3d(verts_obj[0]);
				Vec3d Ngeomd_obj = glm::cross(ud,vd);

				double lsqd = glm::dot(Ngeomd_obj,Ngeomd_obj);
				if (lsqd>0.000000001) {
					Ngeomd_obj *= rsqrt(lsqd);
				} else if (lsqd>0.0) {
					Ngeomd_obj /= std::sqrt(lsqd);
					Ngeomd_obj = glm::normalize(Ngeomd_obj);
				} else {
					//TODO: even more fallbacks?  E.g. try to make it at-least perpendicular to u or
					//	v?
					Ngeomd_obj = Vec3d(1.0,0.0,0.0);
				}

				Ngeomf_obj = Vec3f(Ngeomd_obj);
			}
			#endif

			//Shading normal
			Vec3f Nshad_obj;
			if (_data->buffers_descriptor.has_norms) {
				Vec3f Nshad0_obj = _data->norms[indices_verts.x];
				Vec3f Nshad1_obj = _data->norms[indices_verts.y];
				Vec3f Nshad2_obj = _data->norms[indices_verts.z];
				Nshad_obj = bary_3D.x*Nshad0_obj + bary_3D.y*Nshad1_obj + bary_3D.z*Nshad2_obj;

				//	Normalize it
				#if 1
				lsqf = glm::dot(Nshad_obj,Nshad_obj);
				if (lsqf>0.00001f) {
					Nshad_obj *= rsqrtf(lsqf);
				} else {
					//TODO: more-sophisticated fallback?
					Nshad_obj = Ngeomf_obj;
				}
				#endif

				//Ensure the normals are on the same side.  Flip the geometry normal if they're
				//	different; most-likely the triangle was wound the wrong way, but the shading normals
				//	are correct.
				if (glm::dot(Ngeomf_obj,Nshad_obj)<0.0f) Ngeomf_obj=-Ngeomf_obj;
			} else {
				Nshad_obj = Ngeomf_obj;
			}

			//Perturb by normalmap
			if (has_texc0s) {
				switch (_material->type) {
					case Scene::MaterialBase::TYPE::METALLIC_ROUGHNESS_RGBA: {
						if (_material->metallic_roughness_rgba.normalmap!=0) {
							Mat3x3f matr_TBN = _get_matr_TBN( verts_obj, texc0s, Ngeomf_obj,Nshad_obj );

							Vec3f normal_tangspace = sample_texture( _material->metallic_roughness_rgba.normalmap, texc0 );
							normal_tangspace = normal_tangspace*2.0f - Vec3f(1.0f);

							Nshad_obj = glm::normalize( matr_TBN * normal_tangspace );
						}
						break;
					}
				}
			}

			//Transform to world space
			Ngeom_wld = from_float3(optixTransformNormalFromObjectToWorldSpace(to_float3( Ngeomf_obj )));
			Nshad_wld = from_float3(optixTransformNormalFromObjectToWorldSpace(to_float3( Nshad_obj  )));

			//Normalize
			//	Shouldn't be necessary, but it seems that OptiX doesn't bother making the transform
			//		conserve length.  TODO: can remove some normalizations above if we're doing it
			//		here instead.
			Ngeom_wld = glm::normalize(Ngeom_wld);
			Nshad_wld = glm::normalize(Nshad_wld);
		}

		__device__ void compute_normals_and_dependencies() {
			compute_bary   ();
			compute_texc0s ();
			compute_verts  ();
			compute_normals();
		}
};


class ShadingOperation final {
	public:
		ShadeInfo shade_info;

		RNG*const rng;

		Vec4f albedo;

		Vec3f w_i;
		Vec3f w_o;
		float pdf;

	public:
		__device__ explicit ShadingOperation(RNG* rng) : rng(rng) {
			shade_info.compute_bary();
			shade_info.compute_texc0s();

			_compute_albedo();
		}
		~ShadingOperation() = default;

	private:
		//Always computed.
		__device__ void _compute_albedo() {
			albedo = shade_info._material->compute_albedo(&shade_info);
		}

	public:
		__device__ void compute_shade_info_normals    () {
			shade_info.compute_verts  ();
			shade_info.compute_normals();
		}
		__device__ void compute_shade_info_pos_normals() {
			compute_shade_info_normals();
			shade_info.compute_pos();
		}

		//Flips and perturbs normals as-necessary so that from the direction `.w_o` it appears like
		//	a surface: either normal dotted with `.w_o` is nonnegative.
		__device__ void fix_normals_from_w_o() {
			//Flip normals
			if (glm::dot(w_o,shade_info.Ngeom_wld)>=0.0f);
			else {
				//Hit backface
				shade_info.Ngeom_wld = -shade_info.Ngeom_wld;
				shade_info.Nshad_wld = -shade_info.Nshad_wld;
			}

			//Geometry normal is now correct, but shading normal can still be on the wrong side in
			//	cases near silhouette edges.  Resolve this by perturbing the normal until it is
			//	correct again.
			float dp = glm::dot(w_o,shade_info.Nshad_wld);
			if (dp<=0.00001f) {
				shade_info.Nshad_wld += (0.00001f-dp) * w_o;
			}
		}

		__device__ Vec3f compute_edf_emission() {
			return shade_info._material->compute_edf_emission(this);
		}
		__device__ Vec4f compute_bsdf_evaluate() {
			//return Vec4f(Vec3f(albedo)*RECIP_PI,albedo.a);
			return shade_info._material->compute_bsdf_evaluate(this);
		}
		__device__ Vec4f compute_bsdf_interact() {
			return shade_info._material->compute_bsdf_interact(this);
		}

		//Requires `.compute_albedo()`.
		__device__ bool stochastic_is_opaque() const {
			if        (albedo.a==1.0f) {
				return true;
			} else if (albedo.a!=0.0f) {
				return rng->get_uniform() <= albedo.a;
			} else {
				return false;
			}
		}
};


}
