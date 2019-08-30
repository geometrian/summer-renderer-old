#include "object.hpp"


namespace Summer { namespace Scene {


Object::Mesh::Mesh(TYPE_PRIMS type_prims) :
	type_prims(type_prims), accel(nullptr)
{
	buffers_descriptor.packed = 0u;

	verts = nullptr;
	norms = nullptr;
	tangs = nullptr;

	texcoords0.u8x2 = nullptr;
	texcoords1.u8x2 = nullptr;

	colors0.u8x3 = nullptr;

	indices.u16 = nullptr;
}
Object::Mesh::~Mesh() {
	delete accel;
}

void Object::Mesh::set_ref_verts(DataBlock::Accessor<Vec3f> const* accessor) {
	verts = accessor;
	buffers_descriptor.has_verts = accessor!=nullptr?0b1u:0b0u;
}
void Object::Mesh::set_ref_norms(DataBlock::Accessor<Vec3f> const* accessor) {
	norms = accessor;
	buffers_descriptor.has_norms = accessor!=nullptr?0b1u:0b0u;
}
void Object::Mesh::set_ref_tangs(DataBlock::Accessor<Vec4f> const* accessor) {
	tangs = accessor;
	buffers_descriptor.has_tangs = accessor!=nullptr?0b1u:0b0u;
}

void Object::Mesh::set_ref_texcoords0(std::nullptr_t                     accessor) {
	texcoords0.u8x2 = accessor;
	buffers_descriptor.type_texcoords0 = 0b00u;
}
void Object::Mesh::set_ref_texcoords0(DataBlock::Accessor<Vec2ub> const* accessor) {
	texcoords0.u8x2 = accessor;
	buffers_descriptor.type_texcoords0 = accessor!=nullptr?0b01:0b00u;
}
void Object::Mesh::set_ref_texcoords0(DataBlock::Accessor<Vec2us> const* accessor) {
	texcoords0.u16x2 = accessor;
	buffers_descriptor.type_texcoords0 = accessor!=nullptr?0b10:0b00u;
}
void Object::Mesh::set_ref_texcoords0(DataBlock::Accessor<Vec2f > const* accessor) {
	texcoords0.f32x2 = accessor;
	buffers_descriptor.type_texcoords0 = accessor!=nullptr?0b11:0b00u;
}
void Object::Mesh::set_ref_texcoords1(std::nullptr_t                     accessor) {
	texcoords1.u8x2 = accessor;
	buffers_descriptor.type_texcoords1 = 0b00u;
}
void Object::Mesh::set_ref_texcoords1(DataBlock::Accessor<Vec2ub> const* accessor) {
	texcoords1.u8x2 = accessor;
	buffers_descriptor.type_texcoords1 = accessor!=nullptr?0b01:0b00u;
}
void Object::Mesh::set_ref_texcoords1(DataBlock::Accessor<Vec2us> const* accessor) {
	texcoords1.u16x2 = accessor;
	buffers_descriptor.type_texcoords1 = accessor!=nullptr?0b10:0b00u;
}
void Object::Mesh::set_ref_texcoords1(DataBlock::Accessor<Vec2f > const* accessor) {
	texcoords1.f32x2 = accessor;
	buffers_descriptor.type_texcoords1 = accessor!=nullptr?0b11:0b00u;
}

void Object::Mesh::set_ref_colors0(std::nullptr_t                     accessor) {
	colors0.u8x3 = accessor;
	buffers_descriptor.type_colors0 = 0b000u;
}
void Object::Mesh::set_ref_colors0(DataBlock::Accessor<Vec3ub> const* accessor) {
	colors0.u8x3 = accessor;
	buffers_descriptor.type_colors0 = accessor!=nullptr?0b001:0b000u;
}
void Object::Mesh::set_ref_colors0(DataBlock::Accessor<Vec3us> const* accessor) {
	colors0.u16x3 = accessor;
	buffers_descriptor.type_colors0 = accessor!=nullptr?0b010:0b000u;
}
void Object::Mesh::set_ref_colors0(DataBlock::Accessor<Vec3f > const* accessor) {
	colors0.f32x3 = accessor;
	buffers_descriptor.type_colors0 = accessor!=nullptr?0b011:0b000u;
}
void Object::Mesh::set_ref_colors0(DataBlock::Accessor<Vec4ub> const* accessor) {
	colors0.u8x4 = accessor;
	buffers_descriptor.type_colors0 = accessor!=nullptr?0b101:0b000u;
}
void Object::Mesh::set_ref_colors0(DataBlock::Accessor<Vec4us> const* accessor) {
	colors0.u16x4 = accessor;
	buffers_descriptor.type_colors0 = accessor!=nullptr?0b110:0b000u;
}
void Object::Mesh::set_ref_colors0(DataBlock::Accessor<Vec4f > const* accessor) {
	colors0.f32x4 = accessor;
	buffers_descriptor.type_colors0 = accessor!=nullptr?0b111:0b000u;
}

void Object::Mesh::set_ref_indices(std::nullptr_t                       accessor) {
	indices.u16 = accessor;
	buffers_descriptor.type_indices = 0b00u;
}
void Object::Mesh::set_ref_indices(DataBlock::Accessor<uint16_t> const* accessor) {
	indices.u16 = accessor;
	buffers_descriptor.type_indices = accessor!=nullptr?0b01u:0b00u;
}
void Object::Mesh::set_ref_indices(DataBlock::Accessor<uint32_t> const* accessor) {
	indices.u32 = accessor;
	buffers_descriptor.type_indices = accessor!=nullptr?0b10u:0b00u;
}

void Object::Mesh::upload(OptiX::Context const* context_optix) {
	assert_term(buffers_descriptor.has_verts!=0u,"Must have at-least vertices in object mesh!");

	_ptrs_vbuffers[0] = verts->get_ptr_gpu().ptr_integral;
	assert_term(_ptrs_vbuffers[0]!=reinterpret_cast<CUdeviceptr>(nullptr),"Implementation error!");

	//TODO: Simplify.
	OptiX::AccelerationStructure::BuilderTriangles builder;
	switch (buffers_descriptor.type_indices) {
		case 0b00:
			builder.add_mesh_triangles_basic(
				{ _ptrs_vbuffers, verts->view->stride,       verts->num_elements       }
			);
			break;
		case 0b01:
			_ptr_ibuffer[0] = indices.u16->get_ptr_gpu().ptr_integral;
			builder.add_mesh_triangles_indexed_u16(
				{ _ptrs_vbuffers, verts->view->stride,       verts->num_elements       },
				{ _ptr_ibuffer,   indices.u16->view->stride, indices.u16->num_elements }
			);
			break;
		case 0b10:
			_ptr_ibuffer[0] = indices.u32->get_ptr_gpu().ptr_integral;
			builder.add_mesh_triangles_indexed_u32(
				{ _ptrs_vbuffers, verts->view->stride,       verts->num_elements       },
				{ _ptr_ibuffer,   indices.u32->view->stride, indices.u32->num_elements }
			);
			break;
		nodefault;
	}
	builder.finish();

	accel = new OptiX::AccelerationStructure(context_optix,builder);
}


Object::Object(std::string const& name) :
	name(name)
{}
Object::~Object() {
	for (Mesh const* mesh : meshes) delete mesh;
}

Object::Mesh* Object::add_new_mesh(Mesh::TYPE_PRIMS type_prims) {
	Mesh* new_mesh = new Mesh(type_prims);
	meshes.emplace_back(new_mesh);
	return new_mesh;
}

void Object::upload(OptiX::Context const* context_optix) {
	for (Mesh* mesh : meshes) mesh->upload(context_optix);
}


}}
