#pragma once


#include "stdafx.hpp"

#include "pointer.hpp"


namespace Summer { namespace CUDA {


class BufferBase;
class Context;


class WritePixelBufferObject2D final {
	public:
		GLuint gl_handle;
		cudaGraphicsResource_t cuda_handle;

		size_t const size;
		size_t const sizeof_texel;

		Pointer<void> mapped_ptr;
	private:
		Context const* _map_context_cuda;

	public:
		WritePixelBufferObject2D(Vec2zu const& res, size_t sizeof_texel);
		~WritePixelBufferObject2D();

		CUdeviceptr   map(Context const* context_cuda);
		void        unmap(                           );

		void copy_to_buffer(Context const* context_cuda, BufferBase* buffer);
};


}}
