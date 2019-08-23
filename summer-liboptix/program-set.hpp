#pragma once


#include "stdafx.hpp"


namespace Summer { namespace OptiX {


class CompiledModule;
class Context;


/*
Encapsulates a (collection of) program(s) required to do a particular operation.  The OptiX API
calls this a "program group".  Note that only one program is in the collection for most cases:
	╔════════════════════╤═════════════════════════╤════════════════════════════╗
	║ Operation          │ Shader(s) in Collection │ Subclass Name              ║
	╠════════════════════╪═════════════════════════╪════════════════════════════╣
	║ Ray Generation     │ Ray Generation          │ ProgramRaygen              ║
	╟────────────────────┼─────────────────────────┼────────────────────────────╢
	║ Miss Callback      │ Miss                    │ ProgramMiss                ║
	╟────────────────────┼─────────────────────────┼────────────────────────────╢
	║ Exception Callback │ Exception               │ ProgramException           ║
	╟────────────────────┼─────────────────────────┼────────────────────────────╢
	║ Hit Operations     │ Closest Hit             │ ProgramsHitOps             ║
	║                    │ Any Hit                 │                            ║
	║                    │ Intersection            │                            ║
	╟────────────────────┼─────────────────────────┼────────────────────────────╢
	║ Callable Functions │ Directly Callable       │ ProgramsCallables          ║
	║                    │ Scheduled Callable      │                            ║
	╚════════════════════╧═════════════════════════╧════════════════════════════╝
*/
class ProgramSetBase {
	public:
		OptixProgramGroup program_set;

	protected:
		ProgramSetBase(Context const* context_optix, OptixProgramGroupDesc const& descr);
	public:
		virtual ~ProgramSetBase();
};

class ProgramRaygen final : public ProgramSetBase {
	private:
		static OptixProgramGroupDesc _get_descr(CompiledModule const* module, char const* entry_function);
	public:
		ProgramRaygen(Context const* context_optix, CompiledModule const* module, char const* entry_function);
		virtual ~ProgramRaygen() = default;
};
class ProgramMiss final : public ProgramSetBase {
	private:
		static OptixProgramGroupDesc _get_descr(CompiledModule const* module, char const* entry_function);
	public:
		ProgramMiss(Context const* context_optix, CompiledModule const* module, char const* entry_function);
		virtual ~ProgramMiss() = default;
};
class ProgramsHitOps final : public ProgramSetBase {
	private:
		static OptixProgramGroupDesc _get_descr(
			CompiledModule const* module_ch, char const* entry_point_ch,
			CompiledModule const* module_ah, char const* entry_point_ah,
			CompiledModule const* module_is, char const* entry_point_is
		);
	public:
		ProgramsHitOps(
			Context const* context_optix,
			CompiledModule const* module_ch, char const* entry_point_ch,
			CompiledModule const* module_ah, char const* entry_point_ah,
			CompiledModule const* module_is, char const* entry_point_is
		);
		virtual ~ProgramsHitOps() = default;
};


}}
