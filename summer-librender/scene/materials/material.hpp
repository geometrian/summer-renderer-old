#pragma once


#include "../../stdafx.hpp"


namespace Summer { namespace Scene {


class ShadePoint;
class ShadePointEvaluate;
class ShadePointInteract;
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

					__device__ Vec4f get_albedo(ShadePoint const* hit) const;
					__device__ Vec3f emission(ShadePoint const* hit) const;
					__device__ Vec4f evaluate(ShadePointEvaluate const* hit) const;
					//__device__ Vec4f interact(ShadePointInteract*       hit) const;
				};
				union {
					MetallicRoughnessRGBA metallic_roughness_rgba;
				};

				__device__ Vec4f get_albedo(ShadePoint const* hit) const;
				__device__ Vec3f emission(ShadePoint const* hit) const;
				__device__ Vec4f evaluate(ShadePointEvaluate const* hit) const;
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


class ShadePoint final {
	public:
		//Note: this is `uint32_t`s because although indices are semantically `size_t`, OptiX does
		//	not support larger index types, and those can only make it slower.
		Vec3u const indices;

		Vec3f const bary;

		Vec2f const texc0;

		MaterialBase::InterfaceGPU const*const material;

		Vec3f const Ngeom;
		Vec3f const Nshad;
};
class ShadePointEvaluate final {
	public:
		ShadePoint point;

		Vec3f const w_i;
		Vec3f const w_o;
};
class ShadePointInteract final {
	public:
		ShadePoint point;

		Vec3f const w_i;
		Vec3f*      w_o;
};


}}
