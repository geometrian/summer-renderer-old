#pragma once

#include "stdafx.hpp"


namespace Summer {


inline static __device__ float3 to_float3  (Vec3f  const& vec) { return { vec.x, vec.y, vec.z }; }
inline static __device__ Vec3f  from_float3(float3 const& vec) { return { vec.x, vec.y, vec.z }; }

template<typename Tout, typename Tin>
inline static __device__ Tout bit_cast(Tin const& value) {
	static_assert(sizeof(Tout)==sizeof(Tin),"Implementation error!");
	Tout result; memcpy(&result,&value,sizeof(Tin));
	return result;
}


#if 0
// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };
  
static __forceinline__ __device__ void *unpackPointer( uint32_t i0, uint32_t i1 ) {
	const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
	void*           ptr = reinterpret_cast<void*>( uptr ); 
	return ptr;
}
static __forceinline__ __device__ void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 ) {
	const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

template<typename T> static __forceinline__ __device__ T *getPRD() { 
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}
#endif

inline static __device__ uint32_t pack_sRGB_A(Vec4f const& srgb_a) {
	Vec4u discrete = Vec4u(glm::clamp( Vec4i(srgb_a * 255.0f), Vec4i(0),Vec4i(255) ));
	return (discrete.a<<24) | (discrete.b<<16) | (discrete.g<<8) | discrete.r;
}


template<typename T>
inline static __device__ T square(T const& value) { return value*value; }


}
