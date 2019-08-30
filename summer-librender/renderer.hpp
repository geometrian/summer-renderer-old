#pragma once


#include "stdafx.hpp"

#include "scene/scenegraph.hpp"


namespace Summer {


class Renderer final {
	public:
		Scene::SceneGraph*const scenegraph;

	private:
		struct {
			CUDA::Device*  device;
			CUDA::Context* context;
		} _cuda;

		struct {
			OptiX::Context* context;

			struct {
				uint32_t max_trace_depth;
				uint32_t max_graph_depth;
				uint32_t max_prim_in_accelstruct;
				uint32_t max_inst_in_accelstruct;
				uint32_t rtcore_version; //Note: `0u` means none
				uint32_t max_instanceid;
				uint32_t numbits_vis_msk;
				uint32_t max_sbt_in_accelstruct;
				uint32_t max_sbt_offset;
			} properties;

			OptiX::Pipeline::Options pipeline_opts;

			std::map<std::string,OptiX::ProgramSetBase*> program_sets;

			OptiX::CompiledModule* module;
		} _optix;

	public:
		class Integrator final {
			public:
				OptiX::ShaderBindingTable* sbt;

				OptiX::Pipeline* pipeline;

			public:
				Integrator(
					Renderer* parent, Scene::SceneGraph* scenegraph,
					            OptiX::ProgramRaygen*          program_raygen,
					std::vector<OptiX::ProgramMiss*   > const& programs_miss,
					std::vector<OptiX::ProgramsHitOps*> const& programsets_hitops
				);
				~Integrator();

				void render(Scene::Scene::InterfaceGPU const& interface_gpu) const;
		};
		std::map<std::string,Integrator*> integrators;

	public:
		explicit Renderer(Scene::SceneGraph* scenegraph);
		~Renderer();

		void render(size_t scene_index, size_t camera_index, float timestamp, std::string const& integrator_name) const;
		//void render_start() {}
		//void render_stop () {}
		//void render_wait () {}
};


}
