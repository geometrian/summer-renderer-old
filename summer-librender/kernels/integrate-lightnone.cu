#pragma once


#include "generic-forward.cu"


namespace Summer {


extern "C" __global__ void __raygen__lightnone() {
	TraceInfoBasic trace_info;
	generic_forward0_raygen<TraceInfoBasic>(trace_info);
}


extern "C" __global__ void __miss__lightnone() {
	generic_forward0_miss<TraceInfoBasic>();
}


extern "C" __global__ void __anyhit__lightnone() {
	TraceInfoBasic const* trace_info = PackedPointer<TraceInfoBasic>::from_payloads01();

	ShadingOperation shade_op(trace_info->rng);

	generic_forward0_anyhit<TraceInfoBasic>(shade_op,trace_info);
}


extern "C" __global__ void __closesthit__lightnone() {
	TraceInfoBasic const* trace_info = PackedPointer<TraceInfoBasic>::from_payloads01();

	ShadingOperation shade_op(trace_info->rng);
	shade_op.compute_shade_info_normals();

	generic_forward0_closesthit<TraceInfoBasic>(shade_op,trace_info);
}


}
