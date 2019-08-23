#pragma once


#include "../stdafx.hpp"

#include "../scene/object.hpp"


namespace Summer {


class DataSBT_Raygen final {
	public:
		void* user_data;
		DataSBT_Raygen() : user_data(nullptr) {}
};


class DataSBT_Miss final {
	public:
		void* user_data;
		DataSBT_Miss() : user_data(nullptr) {}
};


class DataSBT_HitOps final {
	public:
		//TODO: make smaller!

		template<typename T> class Attribute final {
			private:
				CUdeviceptr _attr;
				size_t _stride;

			public:
				explicit Attribute(Scene::DataBlock::Accessor<T> const* accessor) :
					_attr  (accessor!=nullptr?accessor->get_ptr_gpu().ptr_integral:reinterpret_cast<CUdeviceptr>(nullptr)),
					_stride(accessor!=nullptr?accessor->view->stride:~size_t(0))
				{
					if (_stride==0) _stride=sizeof(T);
				}
				~Attribute() = default;

				__device__ bool is_valid() const {
					return _attr!=reinterpret_cast<CUdeviceptr>(nullptr);
				}
				__device__ T const& operator[](size_t index) const {
					return *reinterpret_cast<T*>(_attr+index*_stride);
				}
		};

		Attribute<Vec3f> verts;
		Attribute<Vec3f> norms;
		Attribute<Vec4f> tangs;

		Attribute<uint16_t> indices_u16;
		Attribute<uint32_t> indices_u32;

	public:
		DataSBT_HitOps(Scene::Object::Mesh const* mesh) :
			verts(mesh->verts), norms(mesh->norms), tangs(mesh->tangs),
			indices_u16(mesh->buffers_descriptor.type_indices==0b01u?mesh->indices.u16:nullptr),
			indices_u32(mesh->buffers_descriptor.type_indices==0b10u?mesh->indices.u32:nullptr)
		{}
		~DataSBT_HitOps() = default;
};


}
