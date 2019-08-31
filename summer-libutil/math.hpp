#pragma once


#include "stdafx.hpp"


namespace std {


template<typename type_num> __device_host__ inline static type_num  copysign1(type_num const& from) {
	return from<0 ? -1 : 1;
}
template<                 > __device_host__ inline static float     copysign1(float    const& from) {
	return std::copysignf(1.0f,from);
}


}
