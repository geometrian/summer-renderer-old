#pragma once


#include "../../kernels/stdafx.cuh"

#include "material.hpp"


namespace Summer { namespace Scene {


__device__ Vec4f MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::get_albedo(ShadePoint const* hit) const {
	Vec4f base_color = base_color_factor;
	if (base_color_texture!=0) base_color*=sample_texture( base_color_texture, hit->texc0 );
	return base_color;
}
__device__ Vec3f MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::emission(ShadePoint const* hit) const {
	Vec3f emission = emission_factor;
	if (emission_texture!=0) emission*=Vec3f(sample_texture( emission_texture, hit->texc0 ));
	return emission;
}
__device__ Vec4f MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::evaluate(ShadePointEvaluate const* hit) const {
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
//__device__ Vec4f MaterialBase::InterfaceGPU::MetallicRoughnessRGBA::interact(ShadePointInteract*       hit) const {}


__device__ Vec4f MaterialBase::InterfaceGPU::get_albedo(ShadePoint const* hit) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.get_albedo(hit);
		default:
			return Vec4f(1,0,1,1);
	}
}
__device__ Vec3f MaterialBase::InterfaceGPU::emission(ShadePoint const* hit) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.emission(hit);
		default:
			return Vec4f(1,0,1,1);
	}
}
__device__ Vec4f MaterialBase::InterfaceGPU::evaluate(ShadePointEvaluate const* hit) const {
	switch (type) {
		case TYPE::METALLIC_ROUGHNESS_RGBA:
			return metallic_roughness_rgba.evaluate(hit);
		default:
			return Vec4f(1,0,1,1);
	}
}


}}
