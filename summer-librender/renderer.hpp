#pragma once


#include "stdafx.hpp"


namespace Summer {


namespace Scene {


class SceneGraph;


}


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

			OptiX::CompiledModule* module;

			OptiX::ProgramRaygen*  program_raygen;
			OptiX::ProgramMiss*    program_miss;
			OptiX::ProgramsHitOps* programs_hitops;

			OptiX::ShaderBindingTable* sbt;

			OptiX::Pipeline* pipeline;
		} _optix;

	public:
		explicit Renderer(Scene::SceneGraph* scenegraph);
		~Renderer();

		void render(size_t scene_index, size_t camera_index, float timestamp) const;
		//void render_start() {}
		//void render_stop () {}
		//void render_wait () {}
};


}
