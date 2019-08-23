#include "compiled-module.hpp"

#include "context.hpp"


namespace Summer { namespace OptiX {


CompiledModule::CompiledModule(Context const* context_optix, Pipeline::Options const& pipeline_opts, std::string const& ptx_str) :
	context_optix(context_optix)
{
	OptixModuleCompileOptions opt_comp_mod;
	opt_comp_mod.maxRegisterCount = 0; //No limit
	#ifdef BUILD_DEBUG
		opt_comp_mod.optLevel   = OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		opt_comp_mod.debugLevel = OptixCompileDebugLevel::       OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
		//opt_comp_mod.debugLevel = OptixCompileDebugLevel::       OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	#else
		opt_comp_mod.optLevel   = OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
		opt_comp_mod.debugLevel = OptixCompileDebugLevel::       OPTIX_COMPILE_DEBUG_LEVEL_NONE;
	#endif

	assert_optix(optixModuleCreateFromPTX(
		context_optix->context,
		&opt_comp_mod, &pipeline_opts.comp,
		ptx_str.c_str(), ptx_str.length(),
		nullptr, nullptr,
		&module
	));
}
CompiledModule::~CompiledModule() {
	optixModuleDestroy(module);
}


}}
