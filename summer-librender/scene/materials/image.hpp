#pragma once


#include "../../stdafx.hpp"


namespace Summer { namespace Scene {


class Image2D final {
	public:
		enum class FORMAT {
			SCALAR_F32,
			VEC2_F32,
			VEC3_F32,

			CIEXYZ_F32,
			CIEXYZ_A_F32,

			lRGB_F32,
			lRGB_A_F32,
			sRGB_U8,
			sRGB_A_U8
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

#define SUMMER_DEF_SPEC(ENUM,FMTINT,FMTDATA,DATATYPE,TYPE)\
	template<> class ImageFormatInfo<Image2D::FORMAT::ENUM> final { public:\
		static GLenum const gl_fmt_int   = FMTINT;\
		static GLenum const gl_fmt_data  = FMTDATA;\
		static GLenum const gl_datatype  = DATATYPE;\
		static size_t const sizeof_texel = sizeof(TYPE);\
		typedef TYPE type;\
	};
SUMMER_DEF_SPEC( SCALAR_F32,   GL_R32F,        GL_RED,  GL_FLOAT,         float  )
SUMMER_DEF_SPEC( VEC2_F32,     GL_RG32F,       GL_RG,   GL_FLOAT,         Vec2f  )
SUMMER_DEF_SPEC( VEC3_F32,     GL_RGB32F,      GL_RGB,  GL_FLOAT,         Vec3f  )
SUMMER_DEF_SPEC( CIEXYZ_F32,   GL_RGB32F,      GL_RGB,  GL_FLOAT,         Vec3f  )
SUMMER_DEF_SPEC( CIEXYZ_A_F32, GL_RGBA32F,     GL_RGBA, GL_FLOAT,         Vec4f  )
SUMMER_DEF_SPEC( lRGB_F32,     GL_RGB32F,      GL_RGB,  GL_FLOAT,         Vec3f  )
SUMMER_DEF_SPEC( lRGB_A_F32,   GL_RGBA32F,     GL_RGBA, GL_FLOAT,         Vec4f  )
SUMMER_DEF_SPEC( sRGB_U8,      GL_SRGB8,       GL_RGB,  GL_UNSIGNED_BYTE, Vec3ub )
SUMMER_DEF_SPEC( sRGB_A_U8,    GL_SRGB8_ALPHA8,GL_RGBA, GL_UNSIGNED_BYTE, Vec4ub )
#undef SUMMER_DEF_SPEC


}}
