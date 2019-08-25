#pragma once


#include "../../stdafx.hpp"

#include "image.hpp"


namespace Summer { namespace Scene {


class Image2D;
class Sampler;


class Texture2D final {
	public:
		Vec2zu const res;

		Image2D::FORMAT fmt;

		enum class TYPE_LOC {
			UNINITIALIZED,
			CPU,
			GPU_CUDAOPTIX,
			GPU_OPENGL
		};
		TYPE_LOC location;

		union {
			void* cpu;
			struct {
				GLuint handle;
			} gpu_gl;
			struct {
				Image2D const* image;
				Sampler const* sampler;
				cudaTextureObject_t handles[2];
			} gpu_cudaoptix;
		} data;

	public:
		explicit Texture2D(Vec2zu const& res);
		~Texture2D();

		template<Image2D::FORMAT fmt> size_t set_cpu          (                                            );
		template<Image2D::FORMAT fmt> void   set_cpu          (void const* data                            );
		                              void   set_gpu_cudaoptix(Image2D const* image, Sampler const* sampler);
		template<Image2D::FORMAT fmt> void   set_gpu_opengl   (                                            );
		template<Image2D::FORMAT fmt> void   set_gpu_opengl   (void const* data                            );

		void copy_pbo_to_opengl(CUDA::WritePixelBufferObject2D const* pbo);

		void upload_cudaoptix();
};


}}
