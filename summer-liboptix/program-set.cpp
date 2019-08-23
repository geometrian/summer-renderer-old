#include "program-set.hpp"

#include "compiled-module.hpp"
#include "context.hpp"


namespace Summer { namespace OptiX {


ProgramSetBase::ProgramSetBase(Context const* context_optix, OptixProgramGroupDesc const& descr) {
	OptixProgramGroupOptions opts;
	memset(&opts,0x00,sizeof(OptixProgramGroupOptions));

	assert_optix(optixProgramGroupCreate( context_optix->context, &descr,1u, &opts, nullptr,nullptr, &program_set ));
}
ProgramSetBase::~ProgramSetBase() {
	assert_optix(optixProgramGroupDestroy(program_set));
}


OptixProgramGroupDesc ProgramRaygen::_get_descr(CompiledModule const* module, char const* entry_function) {
	OptixProgramGroupDesc descr;
	memset(&descr,0x00,sizeof(OptixProgramGroupDesc));
	descr.kind                     = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	descr.flags                    = OptixProgramGroupFlags::OPTIX_PROGRAM_GROUP_FLAGS_NONE;
	descr.raygen.module            = module->module;
	descr.raygen.entryFunctionName = entry_function;
	return descr;
}
ProgramRaygen::ProgramRaygen(Context const* context_optix, CompiledModule const* module, char const* entry_function) :
	ProgramSetBase(context_optix,ProgramRaygen::_get_descr(module,entry_function))
{}


OptixProgramGroupDesc ProgramMiss::_get_descr(CompiledModule const* module, char const* entry_function) {
	OptixProgramGroupDesc descr;
	memset(&descr,0x00,sizeof(OptixProgramGroupDesc));
	descr.kind                   = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS;
	descr.flags                  = OptixProgramGroupFlags::OPTIX_PROGRAM_GROUP_FLAGS_NONE;
	descr.miss.module            = module->module;
	descr.miss.entryFunctionName = entry_function;
	return descr;
}
ProgramMiss::ProgramMiss(Context const* context_optix, CompiledModule const* module, char const* entry_function) :
	ProgramSetBase(context_optix,ProgramMiss::_get_descr(module,entry_function))
{}


OptixProgramGroupDesc ProgramsHitOps::_get_descr(
	CompiledModule const* module_ch, char const* entry_point_ch,
	CompiledModule const* module_ah, char const* entry_point_ah,
	CompiledModule const* module_is, char const* entry_point_is
) {
	OptixProgramGroupDesc descr;
	memset(&descr,0x00,sizeof(OptixProgramGroupDesc));
	descr.kind  = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	descr.flags = OptixProgramGroupFlags::OPTIX_PROGRAM_GROUP_FLAGS_NONE;
	if (module_ch!=nullptr) {
		descr.hitgroup.moduleCH            = module_ch->module;
		descr.hitgroup.entryFunctionNameCH = entry_point_ch;
	}
	if (module_ah!=nullptr) {
		descr.hitgroup.moduleAH            = module_ah->module;
		descr.hitgroup.entryFunctionNameAH = entry_point_ah;
	}
	if (module_is!=nullptr) {
		descr.hitgroup.moduleIS            = module_is->module;
		descr.hitgroup.entryFunctionNameIS = entry_point_is;
	}
	return descr;
}
ProgramsHitOps::ProgramsHitOps(
	Context const* context_optix,
	CompiledModule const* module_ch, char const* entry_point_ch,
	CompiledModule const* module_ah, char const* entry_point_ah,
	CompiledModule const* module_is, char const* entry_point_is
) :
	ProgramSetBase(context_optix,ProgramsHitOps::_get_descr(module_ch,entry_point_ch,module_ah,entry_point_ah,module_is,entry_point_is))
{
	assert_term(
		(module_ch==nullptr) == (entry_point_ch==nullptr) &&
		(module_ah==nullptr) == (entry_point_ah==nullptr) &&
		(module_is==nullptr) == (entry_point_is==nullptr),
		"Modules and entry-points should agree on whether they are passed!"
	);
}


}}
