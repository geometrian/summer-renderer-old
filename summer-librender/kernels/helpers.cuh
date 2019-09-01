#pragma once


#include "stdafx.cuh"


namespace Summer {


inline static __device__ float3 to_float3  (Vec3f  const& vec) { return { vec.x, vec.y, vec.z }; }
inline static __device__ Vec3f  from_float3(float3 const& vec) { return { vec.x, vec.y, vec.z }; }


template<typename Tout, typename Tin>
inline static __device__ Tout bit_cast(Tin const& value) {
	static_assert(sizeof(Tout)==sizeof(Tin),"Implementation error!");
	Tout result; memcpy(&result,&value,sizeof(Tin));
	return result;
}


template<typename T> class PackedPointer final {
	static_assert(sizeof(T*)==2*sizeof(uint32_t),"Implementation error!");
	private:
		uint32_t _u[2];

	public:
		PackedPointer() = default;
		__device__ PackedPointer(uint32_t u0, uint32_t u1) { _u[0]=u0; _u[1]=u1; }
		__device__ PackedPointer(T* ptr) { memcpy(_u,&ptr,sizeof(T*)); }
		~PackedPointer() = default;

		__device__ static PackedPointer from_payloads01() {
			return PackedPointer(optixGetPayload_0(),optixGetPayload_1());
		}

		__device__ operator T*() const {
			T* result; memcpy(&result,_u,sizeof(T*)); return result;
		}

		__device__ uint32_t const& operator[](size_t index) const { return _u[index]; }
		__device__ uint32_t&       operator[](size_t index)       { return _u[index]; }
};


/*inline static __device__ uint32_t pack_sRGB_A(Vec4f const& srgb_a) {
	Vec4u discrete = Vec4u(glm::clamp( Vec4i(srgb_a * 255.0f), Vec4i(0),Vec4i(255) ));
	return (discrete.a<<24) | (discrete.b<<16) | (discrete.g<<8) | discrete.r;
}*/


inline static __device__ Vec2f semiAtomicAdd(Vec2f* address, Vec2f const& val) {
	float* address_floats = reinterpret_cast<float*>(address);
	return Vec2f(
		atomicAdd(address_floats,  val[0]),
		atomicAdd(address_floats+1,val[1])
	);
}
inline static __device__ Vec3f semiAtomicAdd(Vec3f* address, Vec3f const& val) {
	float* address_floats = reinterpret_cast<float*>(address);
	return Vec3f(
		atomicAdd(address_floats,  val[0]),
		atomicAdd(address_floats+1,val[1]),
		atomicAdd(address_floats+2,val[2])
	);
}
inline static __device__ Vec4f semiAtomicAdd(Vec4f* address, Vec4f const& val) {
	float* address_floats = reinterpret_cast<float*>(address);
	return Vec4f(
		atomicAdd(address_floats,  val[0]),
		atomicAdd(address_floats+1,val[1]),
		atomicAdd(address_floats+2,val[2]),
		atomicAdd(address_floats+3,val[3])
	);
}


template<typename T>
inline static __device__ T square(T const& value) { return value*value; }


//Offsets ray from surface in a fairly robust way.  See "A Fast and Robust Method for Avoiding Self-
//	Intersection".
//	Note: the normal should be on the same side of the surface as the outgoing ray.
inline static __device__ void offset_ray_orig(Ray* ray, Vec3f const& Ngeom) {
	static_assert(sizeof(float)==sizeof(int),"Implementation error!");

	Vec3i of_i = Vec3i( 256.0f * Ngeom );

	Vec3i pos_i; memcpy( &pos_i,&ray->orig, 3*sizeof(float) );
	pos_i = Vec3i(
		pos_i.x + ( ray->orig.x<0.0f ? -of_i.x : of_i.x ),
		pos_i.y + ( ray->orig.y<0.0f ? -of_i.y : of_i.y ),
		pos_i.z + ( ray->orig.z<0.0f ? -of_i.z : of_i.z )
	);
	Vec3f p_i; memcpy( &p_i,&pos_i, 3*sizeof(int) );

	Vec3f p_abs = glm::abs(ray->orig);
	ray->orig = Vec3f(
		p_abs.x<(1.0f/32.0f) ? ray->orig.x+(1.0f/65536.0f)*Ngeom.x : p_i.x,
		p_abs.y<(1.0f/32.0f) ? ray->orig.y+(1.0f/65536.0f)*Ngeom.y : p_i.y,
		p_abs.z<(1.0f/32.0f) ? ray->orig.z+(1.0f/65536.0f)*Ngeom.z : p_i.z
	);
}


}
