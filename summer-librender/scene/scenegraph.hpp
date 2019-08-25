#pragma once


#include "../stdafx.hpp"

#include "camera.hpp"
#include "datablock.hpp"


namespace Summer { namespace Scene {


class Image2D;
class MaterialBase;
class Sampler;
class Texture2D;

class Object;
class Scene;
class SceneGraph;


class Node final {
	friend class Scene;
	public:
		std::vector<Node*> children;

		class Transform final {
			public:
				enum TYPE {
					NONE          = 0b0'000,
					MSK_TRANSLATE = 0b0'001,
					MSK_ROTATE    = 0b0'010,
					MSK_SCALE     = 0b0'100,
					MATRIX        = 0b1'000
				} type;

				union {
					Mat4x4f matrix;
					struct { Vec3f translate; Vec4f rotate; Vec3f scale; };
				};

			public:
				Transform() : type(TYPE::NONE) {}
				~Transform() = default;
		} transform;

		std::vector<Object*> objects;

	public:
		Node() = default;
		~Node() = default;

	private:
		void _register_for_build(
			OptiX::AccelerationStructure::BuilderInstances* builder,
			Mat4x4f const& transform_wld_to_outside
		) const;
};


class Scene final {
	public:
		SceneGraph const*const parent;

		std::vector<Node*> root_nodes;

		std::vector<Camera*> cameras;

		OptiX::AccelerationStructure* accel;

		class InterfaceGPU final {
			public:
				Camera::InterfaceGPU camera;

				CUdeviceptr materials;

				OptixTraversableHandle traversable;
		};

	public:
		explicit Scene(SceneGraph const* parent);
		~Scene();

		void upload(OptiX::Context const* context_optix);

		InterfaceGPU get_interface(size_t camera_index) const;
};


class SceneGraph final {
	public:
		std::vector<DataBlock*              > datablocks;
		std::vector<DataBlock::View*        > datablock_views;
		std::vector<DataBlock::AccessorBase*> datablock_accessors;

		std::vector<Sampler*     > samplers;
		std::vector<Image2D*     > images;
		std::vector<Texture2D*   > textures;
		std::vector<MaterialBase*> materials;
		CUDA::BufferGPUManaged* materials_gpu;

		std::vector<Object*> objects;

		std::vector<Node*> nodes;

		std::vector<Scene*> scenes;

		std::vector<Camera*> cameras;

	public:
		SceneGraph();
		~SceneGraph();

		void upload(OptiX::Context const* context_optix);
};


}}
