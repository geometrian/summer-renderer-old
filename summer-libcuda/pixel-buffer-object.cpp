#include "pixel-buffer-object.hpp"

#include "buffer.hpp"
#include "context.hpp"


namespace Summer { namespace CUDA {


WritePixelBufferObject2D::WritePixelBufferObject2D(Vec2zu const& res, size_t sizeof_texel) :
	size(res[1]*res[0]*sizeof_texel),
	sizeof_texel(sizeof_texel)
{
	glGenBuffers(1, &gl_handle);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_handle);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size,nullptr, GL_STREAM_COPY);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	assert_cuda(cudaGraphicsGLRegisterBuffer(
		&cuda_handle, gl_handle, cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsWriteDiscard
	));

	DEBUG_ONLY(_map_context_cuda = nullptr;)
}
WritePixelBufferObject2D::~WritePixelBufferObject2D() {
	cudaGraphicsUnregisterResource(cuda_handle);

	glDeleteBuffers(1,&gl_handle);
}

CUdeviceptr WritePixelBufferObject2D::  map(Context const* context_cuda) {
	assert_term(_map_context_cuda==nullptr,"Already mapped!");
	_map_context_cuda = context_cuda;

	assert_cuda(cudaGraphicsMapResources( 1,&cuda_handle, _map_context_cuda->stream ));

	assert_cuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mapped_ptr.ptr),nullptr,cuda_handle));

	return mapped_ptr.ptr_integral;
}
void        WritePixelBufferObject2D::unmap(                           ) {
	assert_term(_map_context_cuda!=nullptr,"Already not mapped!");

	assert_cuda(cudaGraphicsUnmapResources( 1,&cuda_handle, _map_context_cuda->stream ));

	DEBUG_ONLY(_map_context_cuda = nullptr;)
}

void WritePixelBufferObject2D::copy_to_buffer(Context const* context_cuda, BufferBase* buffer) {
	CUDA::BufferGPUWrapper tmp(size,map(context_cuda));
	*buffer = tmp;
	unmap();
}


}}
