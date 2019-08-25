#include "sbt-entries.hpp"


namespace Summer {


template<typename T> DataSBT_HitOps::Attribute<T>& DataSBT_HitOps::Attribute<T>::operator=(Scene::DataBlock::Accessor<T> const* accessor) {
	if (accessor==nullptr) {
		_attr   = reinterpret_cast<CUdeviceptr>(nullptr);
		_stride = ~size_t(0);
	} else {
		_attr   = accessor->get_ptr_gpu().ptr_integral;
		_stride = accessor->view->stride;
		if (_stride==0) _stride=sizeof(T);
	}
	return *this;
}

template class DataSBT_HitOps::Attribute<uint16_t>;
template class DataSBT_HitOps::Attribute<uint32_t>;

template class DataSBT_HitOps::Attribute<Vec2ub>;
template class DataSBT_HitOps::Attribute<Vec2us>;
template class DataSBT_HitOps::Attribute<Vec2f >;

template class DataSBT_HitOps::Attribute<Vec4ub>;
template class DataSBT_HitOps::Attribute<Vec4us>;
template class DataSBT_HitOps::Attribute<Vec4f >;


//#ifdef BUILD_DEBUG
//static size_t sbtentry_count = 0;
//#endif

DataSBT_HitOps::DataSBT_HitOps(Scene::Object::Mesh const* mesh) :
	buffers_descriptor(mesh->buffers_descriptor)
{
	verts = mesh->verts;
	norms = mesh->norms;
	tangs = mesh->tangs;

	switch (buffers_descriptor.type_texcoords0) {
		case 0b00u:                                          break;
		case 0b01u: texcoords0.u8x2 =mesh->texcoords0.u8x2;  break;
		case 0b10u: texcoords0.u16x2=mesh->texcoords0.u16x2; break;
		case 0b11u: texcoords0.f32x2=mesh->texcoords0.f32x2; break;
		nodefault;
	}
	switch (buffers_descriptor.type_texcoords1) {
		case 0b00u:                                          break;
		case 0b01u: texcoords1.u8x2 =mesh->texcoords1.u8x2;  break;
		case 0b10u: texcoords1.u16x2=mesh->texcoords1.u16x2; break;
		case 0b11u: texcoords1.f32x2=mesh->texcoords1.f32x2; break;
		nodefault;
	}

	//TODO: colors

	switch (buffers_descriptor.type_indices) {
		case 0b00u:                                break;
		case 0b01u: indices.u16=mesh->indices.u16; break;
		case 0b10u: indices.u32=mesh->indices.u32; break;
		nodefault;
	}

	material_index = mesh->material_index;

	//sbtentry_index = sbtentry_count++;
}


}
