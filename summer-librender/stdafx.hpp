#pragma once


//#include "../summer-libbase/include.hpp" //`#include`d by below
//#include "../summer-libcuda/include.hpp" //`#include`d by below
#include "../summer-libopengl/include.hpp"
#include "../summer-liboptix/include.hpp"
#include "../summer-libutil/include.hpp"

#include <cmath>


#define SUMMER_MAX_RAYTYPES 2
#define SUMMER_SAMPLES_PER_FRAME 1


class Ray final {
	public:
		Vec3f orig;
		Vec3f dir;

		__device__ Vec3f at(float distance) const {
			return orig + distance*dir;
		}
};
