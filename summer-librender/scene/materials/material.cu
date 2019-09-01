#pragma once


#include "material.hpp"

#include "../../kernels/helpers.cuh"
#include "../../kernels/shading.cuh"


namespace Summer { namespace Scene {


__device__ Vec4f MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::compute_albedo(ShadeInfo const* shade_info) const {
	Vec4f base_color = base_color_factor;
	if (base_color_texture!=0) base_color*=sample_texture( base_color_texture, shade_info->texc0 );
	return base_color;
}
__device__ Vec3f MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::compute_edf_emission(ShadingOperation const* shade_op) const {
	Vec3f emission = emission_factor;
	if (emission_texture!=0) emission*=Vec3f(sample_texture( emission_texture, shade_op->shade_info.texc0 ));
	return emission;
}
__device__ Vec4f MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::compute_bsdf_evaluate(ShadingOperation const* shade_op) const {
	Vec3f const& N = shade_op->shade_info.Nshad_wld;
	Vec3f const& w_i = shade_op->w_i;
	Vec3f const& w_o = shade_op->w_o;

	float NdotL = glm::dot(N,w_i);
	if (NdotL>0.0f);
	else return Vec4f(Vec3f(0.0f),shade_op->albedo.a);

	//https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-b-brdf-implementation

	//Load basic BRDF parameters
	Vec4f base_color;
	float metallic, roughness;
	{
		//base_color = base_color_factor;
		//if (base_color_texture!=0) base_color*=sample_texture( base_color_texture, shade_op->shade_info.texc0 );
		base_color = shade_op->albedo; //Computed already

		metallic = metallic_factor;
		if (metallic_texture!=0) metallic*=sample_texture( metallic_texture, shade_op->shade_info.texc0 ).x;

		roughness = roughness_factor;
		if (roughness_texture!=0) roughness*=sample_texture( roughness_texture, shade_op->shade_info.texc0 ).y;
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
	//roughness = glm::clamp(roughness,0.1f,1.0f);
	Vec3f H;
	float alpha, alpha_sq;
	{
		H = glm::normalize( w_i + w_o );

		alpha = roughness * roughness;
		alpha_sq = alpha*alpha;
	}

	//Calculate Fresnel
	Vec3f F = F_0 + (Vec3f(1.0f)-F_0)*std::powf( glm::clamp(1.0f-glm::dot(w_o,H),0.0f,1.0f), 5.0f );

	//Calculate diffuse component (Lambertian)
	Vec3f f_diffuse;
	{
		Vec3f diffuse = c_diff * RECIP_PI;
		f_diffuse = (Vec3f(1.0f)-F) * diffuse;
	}

	//Calculate specular component (microfacet)
	Vec3f f_specular;
	{
		      NdotL = glm::clamp(NdotL,          0.0f,1.0f);
		float NdotV = glm::clamp(glm::dot(N,w_o),0.0f,1.0f);
		float NdotH = glm::clamp(glm::dot(N,H  ),0.0f,1.0f);
		float LdotH = glm::clamp(glm::dot(w_i,H),0.0f,1.0f);

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
	//f_specular = Vec3f(0.0f);

	Vec4f f = Vec4f( f_diffuse + f_specular, base_color.a );
	return f;
}
__device__ Vec4f MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::compute_bsdf_interact(ShadingOperation*       shade_op) const {
	//TODO: this!
	return Vec4f(1,0,1,1);
}


__device__ Vec4f MaterialBase::InterfaceGPU::compute_albedo(ShadeInfo const* shade_info) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.compute_albedo(shade_info);
		default:
			return Vec4f( 1.0f,0.0f,1.0f, 1.0f );
	}
}
__device__ Vec3f MaterialBase::InterfaceGPU::compute_edf_emission(ShadingOperation const* shade_op) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.compute_edf_emission(shade_op);
		default:
			return Vec4f(1,0,1,1);
	}
}
__device__ Vec4f MaterialBase::InterfaceGPU::compute_bsdf_evaluate(ShadingOperation const* shade_op) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.compute_bsdf_evaluate(shade_op);
		default:
			return Vec4f(1,0,1,1);
	}
}
__device__ Vec4f MaterialBase::InterfaceGPU::compute_bsdf_interact(ShadingOperation*       shade_op) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.compute_bsdf_interact(shade_op);
		default:
			return Vec4f(1,0,1,1);
	}
}


}}
