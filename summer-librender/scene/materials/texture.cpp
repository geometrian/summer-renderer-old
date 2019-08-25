#include "texture.hpp"

#include "sampler.hpp"


namespace Summer { namespace Scene {


Texture2D::Texture2D(Vec2zu const& res) :
	res(res), location(TYPE_LOC::UNINITIALIZED)
{}
Texture2D::~Texture2D() {
	switch (location) {
		case TYPE_LOC::UNINITIALIZED:
			break;
		case TYPE_LOC::CPU:
			delete[] static_cast<uint8_t*>(data.cpu);
			break;
		case TYPE_LOC::GPU_CUDAOPTIX:
			if (data.gpu_cudaoptix.handles[0]!=0) {
				assert_cuda(cudaDestroyTextureObject( data.gpu_cudaoptix.handles[0] ));
				assert_cuda(cudaDestroyTextureObject( data.gpu_cudaoptix.handles[1] ));
			}
			break;
		case TYPE_LOC::GPU_OPENGL:
			glDeleteTextures(1,&data.gpu_gl.handle);
			break;
		nodefault;
	}
}

template<Image2D::FORMAT fmt> size_t Texture2D::set_cpu          (                                            ) {
	assert_term(location==TYPE_LOC::UNINITIALIZED,"Already has data!");

	size_t size = res[1] * res[0] * ImageFormatInfo<fmt>::sizeof_texel;
	this->data.cpu = new uint8_t[size];

	this->fmt = fmt;
	location  = TYPE_LOC::CPU;

	return size;
}
template<Image2D::FORMAT fmt> void   Texture2D::set_cpu          (void const* data                            ) {
	size_t size = set_cpu<fmt>();
	memcpy( this->data.cpu,data, size );
}
                              void   Texture2D::set_gpu_cudaoptix(Image2D const* image, Sampler const* sampler) {
	data.gpu_cudaoptix.image   = image;
	data.gpu_cudaoptix.sampler = sampler;
	data.gpu_cudaoptix.handles[0] = 0;
	data.gpu_cudaoptix.handles[1] = 0;

	fmt      = image->format;
	location = TYPE_LOC::GPU_CUDAOPTIX;
}
template<Image2D::FORMAT fmt> void   Texture2D::set_gpu_opengl   (                                            ) {
	glGenTextures(1,&data.gpu_gl.handle);
	glBindTexture(GL_TEXTURE_2D, data.gpu_gl.handle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		ImageFormatInfo<fmt>::gl_fmt_int,
		static_cast<GLsizei>(res[0]), static_cast<GLsizei>(res[1]),
		0,
		ImageFormatInfo<fmt>::gl_fmt_data, ImageFormatInfo<fmt>::gl_datatype, nullptr
	);
	glBindTexture(GL_TEXTURE_2D, 0);

	this->fmt = fmt;
	location  = TYPE_LOC::GPU_OPENGL;
}
template<Image2D::FORMAT fmt> void   Texture2D::set_gpu_opengl   (void const* data                            ) {
	set_gpu_opengl<fmt>();

	glBindTexture(GL_TEXTURE_2D, this->data.gpu_gl.handle);
	glTexSubImage2D(
		GL_TEXTURE_2D,
		0,
		0, 0, static_cast<GLsizei>(res[0]), static_cast<GLsizei>(res[1]),
		ImageFormatInfo<fmt>::gl_fmt_data, ImageFormatInfo<fmt>::gl_datatype, nullptr
	);
	glBindTexture(GL_TEXTURE_2D, 0);
}

#define SUMMER_INSTANTIATE(FMT)\
	template size_t Texture2D::set_cpu       <FMT>(                );\
	template void   Texture2D::set_cpu       <FMT>(void const* data);\
	template void   Texture2D::set_gpu_opengl<FMT>(                );\
	template void   Texture2D::set_gpu_opengl<FMT>(void const* data);
SUMMER_INSTANTIATE(Image2D::FORMAT::sRGB8   )
SUMMER_INSTANTIATE(Image2D::FORMAT::sRGB8_A8)
SUMMER_INSTANTIATE(Image2D::FORMAT::DEPTH32F)
SUMMER_INSTANTIATE(Image2D::FORMAT::lRGB32F )
SUMMER_INSTANTIATE(Image2D::FORMAT::lRGBA32F)
#undef SUMMER_INSTANTIATE

void Texture2D::copy_pbo_to_opengl(CUDA::WritePixelBufferObject2D const* pbo) {
	assert_term(location==TYPE_LOC::GPU_OPENGL,"Must be an OpenGL texture!");

	glBindTexture(GL_TEXTURE_2D, data.gpu_gl.handle);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->gl_handle);
 
	GLenum gl_fmt_data, gl_datatype;
	switch (fmt) {
		#define SUMMER_CASE(FMT)\
			case FMT:\
				gl_fmt_data = ImageFormatInfo<FMT>::gl_fmt_data;\
				gl_datatype = ImageFormatInfo<FMT>::gl_datatype;\
				break;
		SUMMER_CASE(Image2D::FORMAT::sRGB8   )
		SUMMER_CASE(Image2D::FORMAT::sRGB8_A8)
		SUMMER_CASE(Image2D::FORMAT::DEPTH32F)
		SUMMER_CASE(Image2D::FORMAT::lRGB32F )
		SUMMER_CASE(Image2D::FORMAT::lRGBA32F)
		#undef SUMMER_CASE
		nodefault;
	}
	glTexSubImage2D(
		GL_TEXTURE_2D,
		0,
		0, 0, static_cast<GLsizei>(res[0]), static_cast<GLsizei>(res[1]),
		gl_fmt_data, gl_datatype, nullptr
	);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture2D::upload_cudaoptix() {
	cudaResourceDesc descr_resource;
	descr_resource.resType = cudaResourceType::cudaResourceTypeArray;
	descr_resource.res.array.array = data.gpu_cudaoptix.image->data.gpu;

	cudaTextureDesc descrs[2];
	data.gpu_cudaoptix.sampler->get_texture_descrs(descrs);

	assert_cuda(cudaCreateTextureObject(
		data.gpu_cudaoptix.handles,
		&descr_resource, descrs,   nullptr
	));
	assert_cuda(cudaCreateTextureObject(
		data.gpu_cudaoptix.handles+1,
		&descr_resource, descrs+1, nullptr
	));
}




}}
