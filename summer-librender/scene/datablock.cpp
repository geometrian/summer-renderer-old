#include "datablock.hpp"


namespace Summer { namespace Scene {


DataBlock::View::View(DataBlock* datablock, size_t offset,size_t stride,size_t size) :
	datablock(datablock),
	offset(offset), stride(stride), size(size)
{
	assert_term(size>0,"Invalid datablock view length!");
	assert_term(offset+size<=datablock->data.cpu->size,"Datablock view outside buffer!");
}


DataBlock::AccessorBase::AccessorBase(TYPE type, View const* view, size_t offset,size_t num_elements) :
	type(type),
	view(view),
	offset(offset), num_elements(num_elements)
{
	assert_term(offset<view->size,"Accessor must have data in buffer view!");
}

uint8_t*               DataBlock::AccessorBase::get_ptr_cpu() const {
	return
		static_cast<uint8_t*>(view->datablock->data.cpu->ptr) +
		                      view->offset +
		                      offset
	;
}
CUDA::Pointer<uint8_t> DataBlock::AccessorBase::get_ptr_gpu() const {
	return
		static_cast<uint8_t*>(view->datablock->data.gpu->ptr) +
		                      view->offset +
		                      offset
	;
}


template<typename T> DataBlock::AccessorBase::TYPE DataBlock::Accessor<T>::_get_type() {
	#define MYENUM DataBlock::AccessorBase::TYPE::

	if      constexpr (std::is_same_v<T, uint8_t>) return MYENUM U8;
	else if constexpr (std::is_same_v<T,  int8_t>) return MYENUM S8;
	else if constexpr (std::is_same_v<T,uint16_t>) return MYENUM U16;
	else if constexpr (std::is_same_v<T, int16_t>) return MYENUM S16;
	else if constexpr (std::is_same_v<T,uint32_t>) return MYENUM U32;
	else if constexpr (std::is_same_v<T, int32_t>) return MYENUM S32;
	else if constexpr (std::is_same_v<T,float   >) return MYENUM F32;
	else if constexpr (std::is_same_v<T,double  >) return MYENUM F64;

	else if constexpr (std::is_same_v<T,Vec2u>) return MYENUM U32x2;
	else if constexpr (std::is_same_v<T,Vec2i>) return MYENUM S32x2;
	else if constexpr (std::is_same_v<T,Vec2f>) return MYENUM F32x2;
	else if constexpr (std::is_same_v<T,Vec2d>) return MYENUM F64x2;

	else if constexpr (std::is_same_v<T,Vec3u>) return MYENUM U32x3;
	else if constexpr (std::is_same_v<T,Vec3i>) return MYENUM S32x3;
	else if constexpr (std::is_same_v<T,Vec3f>) return MYENUM F32x3;
	else if constexpr (std::is_same_v<T,Vec3d>) return MYENUM F64x3;

	else if constexpr (std::is_same_v<T,Vec4u>) return MYENUM U32x4;
	else if constexpr (std::is_same_v<T,Vec4i>) return MYENUM S32x4;
	else if constexpr (std::is_same_v<T,Vec4f>) return MYENUM F32x4;
	else if constexpr (std::is_same_v<T,Vec4d>) return MYENUM F64x4;

	else if constexpr (std::is_same_v<T,Mat2x2f>) return MYENUM F32x2x2;
	else if constexpr (std::is_same_v<T,Mat2x2d>) return MYENUM F64x2x2;

	else if constexpr (std::is_same_v<T,Mat3x3f>) return MYENUM F32x3x3;
	else if constexpr (std::is_same_v<T,Mat3x3d>) return MYENUM F64x3x3;

	else if constexpr (std::is_same_v<T,Mat4x4f>) return MYENUM F32x4x4;
	else if constexpr (std::is_same_v<T,Mat4x4d>) return MYENUM F64x4x4;

	else static_assert(sizeof(T)==0,"Not implemented!");

	#undef MYENUM
}

template class DataBlock::Accessor< uint8_t>;
template class DataBlock::Accessor<  int8_t>;
template class DataBlock::Accessor<uint16_t>;
template class DataBlock::Accessor< int16_t>;
template class DataBlock::Accessor<uint32_t>;
template class DataBlock::Accessor< int32_t>;
template class DataBlock::Accessor<float   >;
template class DataBlock::Accessor<double  >;

template class DataBlock::Accessor<Vec2u>;
template class DataBlock::Accessor<Vec2i>;
template class DataBlock::Accessor<Vec2f>;
template class DataBlock::Accessor<Vec2d>;

template class DataBlock::Accessor<Vec3u>;
template class DataBlock::Accessor<Vec3i>;
template class DataBlock::Accessor<Vec3f>;
template class DataBlock::Accessor<Vec3d>;

template class DataBlock::Accessor<Vec4u>;
template class DataBlock::Accessor<Vec4i>;
template class DataBlock::Accessor<Vec4f>;
template class DataBlock::Accessor<Vec4d>;

template class DataBlock::Accessor<Mat2x2f>;
template class DataBlock::Accessor<Mat2x2d>;

template class DataBlock::Accessor<Mat3x3f>;
template class DataBlock::Accessor<Mat3x3d>;

template class DataBlock::Accessor<Mat4x4f>;
template class DataBlock::Accessor<Mat4x4d>;


DataBlock::DataBlock(std::vector<uint8_t> const& data) {
	this->data.cpu = new CUDA::BufferCPUManaged(data);
	this->data.gpu = nullptr;
}
DataBlock::~DataBlock() {
	delete data.gpu;
	delete data.cpu;
}

void DataBlock::upload() {
	delete data.gpu;
	data.gpu = new CUDA::BufferGPUManaged(*data.cpu);
}


}}
