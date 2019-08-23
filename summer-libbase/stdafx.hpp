#pragma once


#if   defined __INTEL_COMPILER || defined __ICC //Note: "__ICC" symbol deprecated
	//Note that check for ICC comes before the check for MSVC; ICC still defines "_MSC_VER".
	#define BUILD_COMPILER_INTEL
#elif defined _MSC_VER
	#define BUILD_COMPILER_MSVC
#elif defined __clang__
	//Note that check for Clang comes before check for GNU; Clang defines "__GUNC__" at least some of the time.
	#define BUILD_COMPILER_CLANG
#elif defined __GNUC__
	#define BUILD_COMPILER_GNU
#else
	#error "No supported compiler detected!"
#endif

#ifdef __CUDACC__
	#define BUILD_COMPILER_NVCC
#endif


#ifdef _DEBUG
	#define BUILD_DEBUG
	#define DEBUG_ONLY(CODE) CODE
	#define RELEASE_ONLY(CODE)
#else
	#define BUILD_RELEASE
	#define DEBUG_ONLY(CODE)
	#define RELEASE_ONLY(CODE) CODE
#endif


#include <cuda_runtime.h>

#define OPTIX_COMPATIBILITY 7
#include <optix.h>
#include <optix_stubs.h>

#ifdef BUILD_COMPILER_MSVC
	#define _CRTDBG_MAP_ALLOC
	#include <cstdlib>
	#include <crtdbg.h>
	#ifdef BUILD_DEBUG
		#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
	#else
		#define DBG_NEW new
	#endif
#endif

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#define GLM_FORCE_SIZE_T_LENGTH
/*#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>*/
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>


#if defined BUILD_COMPILER_MSVC || defined BUILD_COMPILER_INTEL
	#define unreachable __assume(0)
#else
	#define unreachable __builtin_unreachable()
#endif

#define COMMA ,


typedef glm::tvec2<uint8_t> Vec2ub;
typedef glm::tvec3<uint8_t> Vec3ub;
typedef glm::tvec4<uint8_t> Vec4ub;

typedef glm::tvec2<uint16_t> Vec2us;
typedef glm::tvec3<uint16_t> Vec3us;
typedef glm::tvec4<uint16_t> Vec4us;

typedef glm::uvec2 Vec2u;
typedef glm::uvec3 Vec3u;
typedef glm::uvec4 Vec4u;

typedef glm::ivec2 Vec2i;
typedef glm::ivec3 Vec3i;
typedef glm::ivec4 Vec4i;

typedef glm::tvec2<size_t> Vec2zu;
typedef glm::tvec3<size_t> Vec3zu;
typedef glm::tvec4<size_t> Vec4zu;

typedef glm::vec2 Vec2f;
typedef glm::vec3 Vec3f;
typedef glm::vec4 Vec4f;

typedef glm::dvec2 Vec2d;
typedef glm::dvec3 Vec3d;
typedef glm::dvec4 Vec4d;

typedef glm::mat2x2 Mat2x2f;
typedef glm::mat3x3 Mat3x3f;
typedef glm::mat4x3 Mat3x4f;
typedef glm::mat4x4 Mat4x4f;

typedef glm::dmat2x2 Mat2x2d;
typedef glm::dmat3x3 Mat3x3d;
typedef glm::dmat4x3 Mat3x4d;
typedef glm::dmat4x4 Mat4x4d;


namespace Summer {


inline static void _message(char const* filename,int line, char const* fmt_cstr,va_list args) {
	fprintf(stderr,"(%s:%d): ",filename,line);
	//Interesting that GNU doesn't notice this warning.  TODO: maybe should tell them about it?
	#ifdef BUILD_COMPILER_CLANG
		#pragma clang diagnostic push
		#pragma clang diagnostic ignored "-Wformat-nonliteral"
	#endif
	vfprintf(stderr, fmt_cstr,args);
	#ifdef BUILD_COMPILER_CLANG
		#pragma clang diagnostic pop
	#endif
	fprintf(stderr,"\n");
}
inline static void _assert_warn (bool pass_condition, char const* filename,int line, char const* fmt_cstr,...) {
	if (pass_condition); else {
		va_list args; va_start(args,fmt_cstr);
		_message(filename,line, fmt_cstr,args);
		va_end(args);
	}
}
inline static void _assert_term (bool pass_condition, char const* filename,int line, char const* fmt_cstr,...) {
	if (pass_condition); else {
		va_list args; va_start(args,fmt_cstr);
		_message(filename,line, fmt_cstr,args);
		va_end(args);
		throw;
	}
}


}
#ifdef BUILD_DEBUG
	#define assert_term(PASS_CONDITION, FMT_CSTR,...)        Summer::_assert_term(PASS_CONDITION, __FILE__,__LINE__, FMT_CSTR,##__VA_ARGS__)
	#define assert_warn(PASS_CONDITION, FMT_CSTR,...)        Summer::_assert_warn(PASS_CONDITION, __FILE__,__LINE__, FMT_CSTR,##__VA_ARGS__)
	#define asserts_term(PASS_EXPR,CHECK_EXPR, FMT_CSTR,...) Summer::_assert_term(PASS_EXPR CHECK_EXPR, __FILE__,__LINE__, FMT_CSTR,##__VA_ARGS__)
	#define asserts_warn(PASS_EXPR,CHECK_EXPR, FMT_CSTR,...) Summer::_assert_warn(PASS_EXPR CHECK_EXPR, __FILE__,__LINE__, FMT_CSTR,##__VA_ARGS__)

	#define implerr Summer::_assert_term(false,__FILE__,__LINE__,"Implementation error!")
	#define notimpl Summer::_assert_term(false,__FILE__,__LINE__,"Not implemented!")
	#define fatalerr(FMT_CSTR,...) Summer::_assert_term(false,__FILE__,__LINE__,FMT_CSTR,##__VA_ARGS__)

	#define assert_optix(CALL) do {\
		OptixResult _call_result = CALL;\
		if (_call_result==OPTIX_SUCCESS); else {\
			fprintf(stderr,"Optix call (%s) (%s:%d) failed (%d)\n",#CALL,__FILE__,__LINE__,_call_result);\
			throw;\
		}\
	} while (0)
	#define assert_cuda(CALL) do {\
		cudaError_t _call_result = CALL;\
		if (_call_result==cudaSuccess); else {\
			fprintf(stderr,"CUDA call (%s) (%s:%d) failed:\n",#CALL,__FILE__,__LINE__);\
			fprintf(stderr,"  %d (%s_:\n",_call_result,cudaGetErrorName(_call_result));\
			fprintf(stderr,"  %s\n",cudaGetErrorString(_call_result));\
			throw;\
		}\
	} while (0)
#else
	#define assert_term(PASS_CONDITION, FMT_CSTR,...)
	#define assert_warn(PASS_CONDITION, FMT_CSTR,...)
	#define asserts_term(PASS_EXPR,CHECK_EXPR, FMT_CSTR,...) PASS_EXPR
	#define asserts_warn(PASS_EXPR,CHECK_EXPR, FMT_CSTR,...) PASS_EXPR

	#define implerr unreachable
	#define notimpl unreachable
	#define fatalerr(FMT_CSTR,...) unreachable

	#define assert_optix(CALL) CALL
	#define assert_cuda(CALL)  CALL
#endif

#ifdef BUILD_DEBUG
	#define nodefault default: implerr
#else
	#define nodefault default: unreachable
#endif
