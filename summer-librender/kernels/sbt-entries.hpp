#pragma once


#include "../stdafx.hpp"

#include "../scene/object.hpp"


namespace Summer {


/*
The structure of the shader binding table for each integrator is fairly straightforward, but still
bears documentation.  Using the example of two objects, each with two meshes, and two raytypes for
the integrator (the actual number of raytypes is `SUMMER_MAX_RAYTYPES`; see below):

	╔════════════════════╤═════════════════════════════╗
	║ Description        │ Shader Table Entries        ║
	╠════════════════════╪═════════════════════════════╣
	║ Ray Generation     │ Raytype 0                   ║
	╟────────────────────┼─────────────────────────────╢
	║ Miss Callback      │ Raytype 0                   ║
	║                    │ Raytype 1                   ║
	╟────────────────────┼─────────────────────────────╢
	║ Hit Operations     │ Object 0, Mesh 0, Raytype 0 ║
	║                    │ Object 0, Mesh 0, Raytype 1 ║
	║                    │ Object 0, Mesh 1, Raytype 0 ║
	║                    │ Object 0, Mesh 1, Raytype 1 ║
	║                    │ Object 1, Mesh 0, Raytype 0 ║
	║                    │ Object 1, Mesh 0, Raytype 1 ║
	║                    │ Object 1, Mesh 1, Raytype 0 ║
	║                    │ Object 1, Mesh 1, Raytype 1 ║
	╚════════════════════╧═════════════════════════════╝

For accessing the hit operations entries, OptiX uses the following formula:

	start + stride*(
		(ray contribution) +
		(geometry multiplier)*(geometry contribution) +
		(instance contribution)
	)

The ray contribution is the ray type.  The geometry contribution is the mesh index, since the lower-
level acceleration structures are built for each object, each containing individual meshes.  The
geometry multiplier is the number of ray types.  The instance contribution is the number of entries
which have appeared before this object.

Note that the order, interleaving the entries by raytype, means that the offset of each object (the
instance contribution) depends on the number of raytypes.  Since this is baked into the upper-level
acceleration structure, a separate upper-level structure must be built for each unique number of ray
types an integrator could need.  This is annoying.  Swapping it around so that the ray contribution
forms the larger jump doesn't work: the ray contribution is required to be much smaller (8 bits vs.
24, it seems).

For simplicity, the largest number of ray types required by any integrator is specified as
`SUMMER_MAX_RAYTYPES`, and then *every* table uses this, with any additional ray types being filled
with zero records.  This is wasteful for the (typically simpler) integrators requiring fewer ray
types, but the implementation is greatly simplified and clarified.
*/


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
