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
		Scene::Object::Mesh::BuffersDescriptor const buffers_descriptor;

		template<typename T> class Attribute final {
			private:
				CUdeviceptr _attr;
				size_t      _stride;

			public:
				Attribute() { *this=nullptr; }
				~Attribute() = default;

				Attribute<T>& operator=(Scene::DataBlock::Accessor<T> const* accessor);

				__device__ T const& operator[](size_t index) const {
					return *reinterpret_cast<T*>(_attr+index*_stride);
				}
		};

		Attribute<Vec3f> verts;
		Attribute<Vec3f> norms;
		Attribute<Vec4f> tangs;

		union TexCoord final {
			Attribute<Vec2ub> u8x2;
			Attribute<Vec2us> u16x2;
			Attribute<Vec2f > f32x2;
			TexCoord() {}
		};
		TexCoord texcoords0;
		TexCoord texcoords1;

		//TODO: colors

		union Indices final {
			Attribute<uint16_t> u16;
			Attribute<uint32_t> u32;
			Indices() {}
		};
		Indices indices;

		size_t material_index;

		//#ifdef BUILD_DEBUG
		//size_t sbtentry_index;
		//#endif

	public:
		explicit DataSBT_HitOps(Scene::Object::Mesh const* mesh);
		~DataSBT_HitOps() = default;
};


}
