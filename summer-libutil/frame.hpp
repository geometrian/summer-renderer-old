#pragma once


#include "stdafx.hpp"

#include "math.hpp"


namespace Summer {


__device_host__ inline static void build_frame(Vec3f const& frame_z, Vec3f*__restrict frame_x,Vec3f*__restrict frame_y) {
	//http://jcgt.org/published/0006/01/01/paper.pdf
	float sign = std::copysign1(frame_z[2]);
	float a = -1.0f / (sign + frame_z[2]);
	float b = frame_z[0] * frame_z[1] * a;
	*frame_x = Vec3f(
		b,
		sign + frame_z[1]*frame_z[1]*a,
		-frame_z[1]
	);
	*frame_y = Vec3f(
		1.0f + sign*frame_z[0]*frame_z[0]*a,
		sign*b,
		-sign*frame_z[0]
	);
}


}
