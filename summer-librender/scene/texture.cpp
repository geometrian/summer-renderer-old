#include "texture.hpp"


namespace Summer {


template<GLint fmt_internal> Texture2D<fmt_internal>::Texture2D(Vec2zu const& res) :
	res(res)
{
	glGenTextures(1,&gl_handle);
	glBindTexture(GL_TEXTURE_2D, gl_handle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		fmt_internal,
		static_cast<GLsizei>(res[0]), static_cast<GLsizei>(res[1]),
		0,
		TextureFormatInfo<fmt_internal>::fmt_data, TextureFormatInfo<fmt_internal>::datatype, nullptr
	);
	glBindTexture(GL_TEXTURE_2D,0);
}
template<GLint fmt_internal> Texture2D<fmt_internal>::~Texture2D() {
	glDeleteTextures(1,&gl_handle);
}

template<GLint fmt_internal> void Texture2D<fmt_internal>::copy_from_pbo(CUDA::WritePixelBufferObject2D const* pbo) {
	glBindTexture(GL_TEXTURE_2D, gl_handle);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo->gl_handle);
 
	glTexSubImage2D(
		GL_TEXTURE_2D,
		0,
		0, 0, static_cast<GLsizei>(res[0]), static_cast<GLsizei>(res[1]),
		TextureFormatInfo<fmt_internal>::fmt_data, TextureFormatInfo<fmt_internal>::datatype, nullptr
	);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

template class Texture2D<GL_R32F  >;
template class Texture2D<GL_RGB8  >;
template class Texture2D<GL_RGB32F>;
template class Texture2D<GL_RGBA8 >;


}
