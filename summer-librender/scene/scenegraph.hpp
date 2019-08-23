#pragma once


#include "../stdafx.hpp"

#include "camera.hpp"
#include "datablock.hpp"


namespace Summer { namespace Scene {


//class Material;
class Object;
class Scene;


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
		std::vector<Node*> root_nodes;

		std::vector<Camera*> cameras;

		OptiX::AccelerationStructure* accel;

		class InterfaceGPU final {
			public:
				Camera::InterfaceGPU camera;

				OptixTraversableHandle traversable;
		};

	public:
		Scene();
		~Scene();

		void upload(OptiX::Context const* context_optix);

		InterfaceGPU get_interface(size_t camera_index) const {
			return { cameras[camera_index]->get_interface(), accel->handle };
		}
};


class SceneGraph final {
	public:
		std::vector<DataBlock*              > datablocks;
		std::vector<DataBlock::View*        > datablock_views;
		std::vector<DataBlock::AccessorBase*> datablock_accessors;

		std::vector<Object*> objects;

		std::vector<Node*> nodes;

		std::vector<Scene*> scenes;

		std::vector<Camera*> cameras;

	public:
		SceneGraph() = default;
		~SceneGraph();

		void upload(OptiX::Context const* context_optix);
};


}}
