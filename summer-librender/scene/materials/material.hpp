#pragma once


#include "../../stdafx.hpp"


namespace Summer {


class ShadeInfo;
class ShadingOperation;


namespace Scene {


class Texture2D;


class MaterialBase {
	public:
		std::string name;

		enum class TYPE {
			METALLIC_ROUGHNESS_RGBA
		};
		TYPE const type;

		class InterfaceGPU final {
			public:
				TYPE type;

				class MetallicRoughnessRGBA final { public:
					Vec4f               base_color_factor;
					Vec3f               emission_factor;
					float               metallic_factor;
					float               roughness_factor;
					cudaTextureObject_t base_color_texture;
					cudaTextureObject_t metallic_texture;
					cudaTextureObject_t roughness_texture;
					cudaTextureObject_t emission_texture;
					cudaTextureObject_t normalmap;

					__device__ Vec4f compute_albedo(ShadeInfo const* shade_info) const;
					__device__ Vec3f compute_edf_emission(ShadingOperation const* shade_op) const;
					__device__ Vec4f compute_bsdf_evaluate(ShadingOperation const* shade_op) const;
					__device__ Vec4f compute_bsdf_interact(ShadingOperation*       shade_op) const;
				};
				union {
					MetallicRoughnessRGBA metallic_roughness_rgba;
				};

				__device__ Vec4f compute_albedo(ShadeInfo const* shade_info) const;
				__device__ Vec3f compute_edf_emission(ShadingOperation const* shade_op) const;
				__device__ Vec4f compute_bsdf_evaluate(ShadingOperation const* shade_op) const;
				__device__ Vec4f compute_bsdf_interact(ShadingOperation*       shade_op) const;
		};

	protected:
		explicit MaterialBase(std::string const& name, TYPE type);
	public:
		virtual ~MaterialBase() = default;

		virtual void fill_interface(InterfaceGPU* interface) const = 0;
};

class MaterialMetallicRoughnessRGBA final : public MaterialBase {
	public:
		class BaseColor final { public:
			Vec4f factor;
			Texture2D const* texture;

			BaseColor() : factor(1.0f,1.0f,1.0f,1.0f),texture(nullptr) {}
		} base_color;
		class Metallic final { public:
			float factor;
			Texture2D const* texture; //Data stored in red channel

			Metallic() : factor(1.0f),texture(nullptr) {}
		} metallic;
		class Roughness final { public:
			float factor;
			Texture2D const* texture; //Data stored in green channel

			Roughness() : factor(1.0f),texture(nullptr) {}
		} roughness;

		class Emissive final { public:
			Vec3f factor;
			Texture2D const* texture;

			Emissive() : factor(1.0f,1.0f,1.0f),texture(nullptr) {}
		} emission;

		class Normal final { public:
			Texture2D const* texture;

			Normal() : texture(nullptr) {}
		} normal;

	public:
		explicit MaterialMetallicRoughnessRGBA(std::string const& name);
		virtual ~MaterialMetallicRoughnessRGBA() = default;

		virtual void fill_interface(InterfaceGPU* interface) const override;
};


}}
