#include "material.hpp"

#include "texture.hpp"


namespace Summer { namespace Scene {


#define SUMMER_OPT_TEX_HANDLE(TEX,INDEX)\
	(TEX!=nullptr) ? TEX->data.gpu_cudaoptix.handles[INDEX] : reinterpret_cast<CUdeviceptr>(nullptr)


MaterialBase::MaterialBase(std::string const& name, TYPE type) :
	name(name), type(type)
{}


MaterialMetallicRoughnessRGBA::MaterialMetallicRoughnessRGBA(std::string const& name) :
	MaterialBase(name,TYPE::METALLIC_ROUGHNESS_RGBA)
{}

void MaterialMetallicRoughnessRGBA::fill_interface(InterfaceGPU* interface) const /*override*/ {
	interface->type = type;

	interface->metallic_roughness_rgba.base_color_factor = base_color.factor;
	interface->metallic_roughness_rgba.emission_factor   = emission.factor;
	interface->metallic_roughness_rgba.metallic_factor   = metallic.factor;
	interface->metallic_roughness_rgba.roughness_factor  = roughness.factor;
	interface->metallic_roughness_rgba.base_color_texture = SUMMER_OPT_TEX_HANDLE(base_color.texture,1);
	interface->metallic_roughness_rgba.metallic_texture   = SUMMER_OPT_TEX_HANDLE(metallic.  texture,1);
	interface->metallic_roughness_rgba.roughness_texture  = SUMMER_OPT_TEX_HANDLE(roughness. texture,1);
	interface->metallic_roughness_rgba.emission_texture   = SUMMER_OPT_TEX_HANDLE(emission.  texture,1);
	interface->metallic_roughness_rgba.normalmap          = SUMMER_OPT_TEX_HANDLE(normal.    texture,0);
}


#undef SUMMER_OPT_TEX_HANDLE


}}
