#pragma once


#include "stdafx.hpp"


namespace Summer { namespace OptiX {


class Context;


class AccelerationStructure final {
	public:
		Context const*const context_optix;

	private:
		class _BuilderBase {
			friend class AccelerationStructure;
			protected:
				std::vector<OptixBuildInput> _build_inputs;
			private:
				DEBUG_ONLY(bool _finished=false;)

			public:
				virtual void finish() = 0;
		};
	public:
		class BuilderTriangles final : public _BuilderBase {
			private:
				std::vector<CUdeviceptr> _verts_ptrs;
				OptixGeometryFlags const _geom_flags[1] = { OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE };
			public:
				class BufferAccessor final { public:
					CUdeviceptr ptr; size_t stride; size_t count;
				};

			public:
				BuilderTriangles() = default;
				~BuilderTriangles() = default;

			private:
				OptixBuildInput& _add_mesh_triangles(BufferAccessor const& vertices);
			public:
				void add_mesh_triangles_basic      (BufferAccessor const& vertices                               );
				void add_mesh_triangles_indexed_u16(BufferAccessor const& vertices, BufferAccessor const& indices);
				void add_mesh_triangles_indexed_u32(BufferAccessor const& vertices, BufferAccessor const& indices);

				virtual void finish() override;
		};
		class BuilderInstances final : public _BuilderBase {
			private:
				std::vector<OptixInstance> _instances;

				CUDA::BufferGPUManaged* _instances_gpu;

			public:
				BuilderInstances();
				~BuilderInstances();

				void add_instance(CUdeviceptr child_traversable, Mat3x4f const& transform);

				virtual void finish() override;
		};

		OptixTraversableHandle handle;
	private:
		CUDA::BufferGPUManaged* _buffer;

	protected:
		void _build(std::vector<OptixBuildInput> const& build_inputs);
	public:
		AccelerationStructure(Context const* context_optix, _BuilderBase const& builder);
		~AccelerationStructure();
};


}}
