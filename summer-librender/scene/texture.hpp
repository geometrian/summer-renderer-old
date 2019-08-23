#pragma once


#include "../stdafx.hpp"


namespace Summer {


template<GLint fmt_internal> class TextureFormatInfo;

template<> class TextureFormatInfo<GL_R32F  > final { public:
	static GLenum const fmt_data = GL_RED;
	static GLenum const datatype = GL_FLOAT;
	static size_t const sizeof_texel = sizeof(float);
};

template<> class TextureFormatInfo<GL_RGB8  > final { public:
	static GLenum const fmt_data = GL_RGB;
	static GLenum const datatype = GL_UNSIGNED_BYTE;
	static size_t const sizeof_texel = 3*sizeof(uint8_t);
};
template<> class TextureFormatInfo<GL_RGB32F> final { public:
	static GLenum const fmt_data = GL_RGB;
	static GLenum const datatype = GL_FLOAT;
	static size_t const sizeof_texel = 3*sizeof(float);
};

template<> class TextureFormatInfo<GL_RGBA8 > final { public:
	static GLenum const fmt_data = GL_RGBA;
	static GLenum const datatype = GL_UNSIGNED_BYTE;
	static size_t const sizeof_texel = 4*sizeof(uint8_t);
};



template<GLint fmt_internal>
class Texture2D final {
	public:
		Vec2zu const res;

		GLuint gl_handle;

	public:
		explicit Texture2D(Vec2zu const& res);
		~Texture2D();

		void copy_from_pbo(CUDA::WritePixelBufferObject2D const* pbo);
};


}
