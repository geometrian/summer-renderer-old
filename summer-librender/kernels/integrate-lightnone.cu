#pragma once


#include "generic-forward.cu"


namespace Summer {


extern "C" __global__ void __raygen__lightnone() {
	generic_forward0_raygen();
}


extern "C" __global__ void __miss__lightnone() {
	generic_forward0_miss();
}


extern "C" __global__ void __anyhit__lightnone() {
	generic_forward0_anyhit();
}


extern "C" __global__ void __closesthit__lightnone() {
	generic_forward0_closesthit();
}


}
