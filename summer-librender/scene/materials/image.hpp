#pragma once


#include "../../stdafx.hpp"


namespace Summer { namespace Scene {


//#define SUMMER_ENABLE_SRGB_OPENGL


class Image2D final {
	public:
		enum class FORMAT {
			sRGB8,
			sRGB8_A8,

			DEPTH32F,
			lRGB32F,
			lRGBA32F,

			VEC3_32F = lRGB32F
		};
		FORMAT const format;

		Vec2zu const res;

		struct {
			std::vector<uint8_t> cpu;
			cudaArray_t gpu;
		} data;

	public:
		explicit Image2D(FORMAT format, Vec2zu const& res);
		~Image2D();

		void upload();
};


template<Image2D::FORMAT fmt> class ImageFormatInfo;

template<> class ImageFormatInfo<Image2D::FORMAT::sRGB8   > final { public:
	#ifdef SUMMER_ENABLE_SRGB_OPENGL
	static GLenum const gl_fmt_int   = GL_SRGB8;
	#else
	static GLenum const gl_fmt_int   = GL_RGB8;
	#endif
	static GLenum const gl_fmt_data  = GL_RGB;
	static GLenum const gl_datatype  = GL_UNSIGNED_BYTE;
	static size_t const sizeof_texel = 3*sizeof(uint8_t);
	typedef Vec3ub type;
};
template<> class ImageFormatInfo<Image2D::FORMAT::sRGB8_A8> final { public:
	#ifdef SUMMER_ENABLE_SRGB_OPENGL
	static GLenum const gl_fmt_int   = GL_SRGB8_ALPHA8;
	#else
	static GLenum const gl_fmt_int   = GL_RGBA8;
	#endif
	static GLenum const gl_fmt_data  = GL_RGBA;
	static GLenum const gl_datatype  = GL_UNSIGNED_BYTE;
	static size_t const sizeof_texel = (3+1)*sizeof(uint8_t);
	typedef Vec4ub type;
};
template<> class ImageFormatInfo<Image2D::FORMAT::DEPTH32F> final { public:
	static GLenum const gl_fmt_int   = GL_R32F;
	static GLenum const gl_fmt_data  = GL_RED;
	static GLenum const gl_datatype  = GL_FLOAT;
	static size_t const sizeof_texel = sizeof(float);
	typedef float type;
};
template<> class ImageFormatInfo<Image2D::FORMAT::lRGB32F > final { public:
	static GLenum const gl_fmt_int   = GL_RGB32F;
	static GLenum const gl_fmt_data  = GL_RGB;
	static GLenum const gl_datatype  = GL_FLOAT;
	static size_t const sizeof_texel = 3*sizeof(float);
	typedef Vec3f type;
};
template<> class ImageFormatInfo<Image2D::FORMAT::lRGBA32F> final { public:
	static GLenum const gl_fmt_int   = GL_RGBA32F;
	static GLenum const gl_fmt_data  = GL_RGBA;
	static GLenum const gl_datatype  = GL_FLOAT;
	static size_t const sizeof_texel = (3+1)*sizeof(float);
	typedef Vec4f type;
};


}}
