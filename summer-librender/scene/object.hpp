#pragma once


#include "../stdafx.hpp"

#include "datablock.hpp"


namespace Summer { namespace Scene {


class MaterialBase;


class Object final {
	public:
		std::string const name;

		class Mesh final {
			friend class Object;
			public:
				enum class TYPE_PRIMS : int {
					POINTS,
					LINES, LINE_LOOP, LINE_STRIP,
					TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN
				} const type_prims;

				//Buffers descriptor: describes which buffers are present (and what type they are,
				//	as-necessary):
				union BuffersDescriptor final {
					struct {
						//Basic attributes
						uint32_t has_verts       :  1; //Must be "1"
						uint32_t has_norms       :  1;
						uint32_t has_tangs       :  1;
						//Texture coordinates:
						//	"00"=none
						//	{"01","10","11"}={`uint8_t`,`uint16_t`,`float`} channels
						uint32_t type_texcoords0 :  2;
						uint32_t type_texcoords1 :  2;
						//Color
						//	"000"=none
						//	"0??"=three-channel, "1??"=four-channel
						//	{"?01","?10","?11"}={`uint8_t`,`uint16_t`,`float`} channels
						uint32_t type_colors0    :  3;
						//TODO: joints
						//TODO: weights
						//Indices
						//	"00"=none
						//	{"01","10"}={16-bit,32-bit}
						uint32_t type_indices    :  2;
						//Padding
						uint32_t                 : 20;
					};
					uint32_t packed;
				} buffers_descriptor;
				static_assert(sizeof(buffers_descriptor)==sizeof(uint32_t),"Implementation error!");

				DataBlock::Accessor<Vec3f> const* verts;
				DataBlock::Accessor<Vec3f> const* norms;
				DataBlock::Accessor<Vec4f> const* tangs;

				union TexCoord final {
					DataBlock::Accessor<Vec2ub> const* u8x2;
					DataBlock::Accessor<Vec2us> const* u16x2;
					DataBlock::Accessor<Vec2f > const* f32x2;
				};
				TexCoord texcoords0;
				TexCoord texcoords1;

				union Color final {
					DataBlock::Accessor<Vec3ub> const* u8x3;
					DataBlock::Accessor<Vec3us> const* u16x3;
					DataBlock::Accessor<Vec3f > const* f32x3;
					DataBlock::Accessor<Vec4ub> const* u8x4;
					DataBlock::Accessor<Vec4us> const* u16x4;
					DataBlock::Accessor<Vec4f > const* f32x4;
				};
				Color colors0;

				union Indices final {
					DataBlock::Accessor<uint16_t> const* u16;
					DataBlock::Accessor<uint32_t> const* u32;
				};
				Indices indices;

				MaterialBase const* material;
				size_t material_index;

				//Lower-level acceleration structure built over primitives
				OptiX::AccelerationStructure* accel;

			private:
				CUdeviceptr mutable _ptrs_vbuffers[1];
				CUdeviceptr mutable _ptr_ibuffer  [1];

			public:
				explicit Mesh(TYPE_PRIMS type_prims);
				~Mesh();

				void set_ref_verts(DataBlock::Accessor<Vec3f> const* accessor);
				void set_ref_norms(DataBlock::Accessor<Vec3f> const* accessor);
				void set_ref_tangs(DataBlock::Accessor<Vec4f> const* accessor);

				void set_ref_texcoords0(std::nullptr_t                     accessor);
				void set_ref_texcoords0(DataBlock::Accessor<Vec2ub> const* accessor);
				void set_ref_texcoords0(DataBlock::Accessor<Vec2us> const* accessor);
				void set_ref_texcoords0(DataBlock::Accessor<Vec2f > const* accessor);
				void set_ref_texcoords1(std::nullptr_t                     accessor);
				void set_ref_texcoords1(DataBlock::Accessor<Vec2ub> const* accessor);
				void set_ref_texcoords1(DataBlock::Accessor<Vec2us> const* accessor);
				void set_ref_texcoords1(DataBlock::Accessor<Vec2f > const* accessor);

				void set_ref_colors0(std::nullptr_t                     accessor);
				void set_ref_colors0(DataBlock::Accessor<Vec3ub> const* accessor);
				void set_ref_colors0(DataBlock::Accessor<Vec3us> const* accessor);
				void set_ref_colors0(DataBlock::Accessor<Vec3f > const* accessor);
				void set_ref_colors0(DataBlock::Accessor<Vec4ub> const* accessor);
				void set_ref_colors0(DataBlock::Accessor<Vec4us> const* accessor);
				void set_ref_colors0(DataBlock::Accessor<Vec4f > const* accessor);

				void set_ref_indices(std::nullptr_t                       accessor);
				void set_ref_indices(DataBlock::Accessor<uint16_t> const* accessor);
				void set_ref_indices(DataBlock::Accessor<uint32_t> const* accessor);

				void upload(OptiX::Context const* context_optix);
		};
		std::vector<Mesh*> meshes;

	public:
		explicit Object(std::string const& name);
		~Object();

		Mesh* add_new_mesh(Mesh::TYPE_PRIMS type_prims);

		void upload(OptiX::Context const* context_optix);
};


}}
