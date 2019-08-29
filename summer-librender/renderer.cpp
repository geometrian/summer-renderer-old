#include "renderer.hpp"

#include "kernels/sbt-entries.hpp"

#include "scene/framebuffer.hpp"
#include "scene/scenegraph.hpp"


extern "C" unsigned char const ptx_embed___summer_librender___kernels___main_cu[];


namespace Summer {


Renderer::Renderer(Scene::SceneGraph* scenegraph) : scenegraph(scenegraph) {
	//CUDA setup
	{
		_cuda.device  = new CUDA::Device(0);
		_cuda.context = new CUDA::Context(_cuda.device);
	}

	//OptiX setup
	{
		_optix.context  = new OptiX::Context(_cuda.context);
	}

	//Upload scenegraph to GPU
	scenegraph->upload(_optix.context);

	//Options for pipeline
	OptiX::Pipeline::Options pipeline_opts;

	//Load shaders, fill binding table, build shading pipeline
	{

		_optix.module = new OptiX::CompiledModule(_optix.context,pipeline_opts,reinterpret_cast<char const*>(ptx_embed___summer_librender___kernels___main_cu));

		_optix.program_raygen  = new OptiX::ProgramRaygen ( _optix.context, _optix.module,"__raygen__primary" );
		_optix.program_miss    = new OptiX::ProgramMiss   ( _optix.context, _optix.module,"__miss__primary"   );
		_optix.programs_hitops = new OptiX::ProgramsHitOps(
			_optix.context,
			_optix.module, "__closesthit__radiance",
			_optix.module, "__anyhit__radiance",
			nullptr,       nullptr
		);

		OptiX::ShaderBindingTable::Builder<DataSBT_Raygen,DataSBT_Miss,DataSBT_HitOps> sbt_builder;
		{
			sbt_builder.raygen = std::make_pair( _optix.program_raygen, new DataSBT_Raygen );

			sbt_builder.miss.emplace_back(std::make_pair( _optix.program_miss, new DataSBT_Miss ));
			//sbt_builder.miss.emplace_back(std::make_pair( _optix.program_miss, new DataSBT_Miss  ));

			#if 1
				for (Scene::Object const* object : scenegraph->objects) {
					for (Scene::Object::Mesh const* mesh : object->meshes) {
						sbt_builder.hitsops.emplace_back(std::make_pair( _optix.programs_hitops, new DataSBT_HitOps(mesh) ));
					}
				}
			#else
				Scene::Scene const* scene = scenegraph->scenes[0];
				for (Scene::Node const* root_node : scene->root_nodes) {
					std::function<void(Scene::Node const*)> add_node = [&](Scene::Node const* node) -> void {
						for (Scene::Node const* child : node->children) {
							add_node(child);
						}
						for (Scene::Object const* object : node->objects) {
							for (Scene::Object::Mesh const* mesh : object->meshes) {
								DataSBT_HitOps entry_hitops(mesh);
								sbt_builder.hitsops.emplace_back(std::make_pair( _optix.programs_hitops, &entry_hitops ));
							}
						}
					};
					add_node(root_node);
				}
			#endif
		}
		_optix.sbt = new OptiX::ShaderBindingTable(sbt_builder);
		                                             delete sbt_builder.raygen.second;
		for (auto const& iter : sbt_builder.miss   ) delete iter.              second;
		for (auto const& iter : sbt_builder.hitsops) delete iter.              second;

		_optix.pipeline = new OptiX::Pipeline( _optix.context, pipeline_opts, _optix.sbt );
	}
}
Renderer::~Renderer() {
	{
		delete _optix.pipeline;

		delete _optix.sbt;

		delete _optix.programs_hitops;
		delete _optix.program_miss;
		delete _optix.program_raygen;

		delete _optix.module;
	}

	{
		delete _optix.context;
	}

	{
		delete _cuda.context;
		delete _cuda.device;
	}
}

void Renderer::render(size_t scene_index, size_t camera_index, float timestamp) const {
	Scene::Scene const* scene = scenegraph->scenes[scene_index];
	Scene::Framebuffer& framebuffer = scene->cameras[camera_index]->framebuffer;

	framebuffer.launch_prepare(_cuda.context);

	Scene::Scene::InterfaceGPU const& interface = scene->get_interface(camera_index);

	//TODO: move outside
	CUDA::BufferGPUManaged launch_interface_buffer( sizeof(Scene::Scene::InterfaceGPU),&interface );
	size_t res[3] = { framebuffer.res.x, framebuffer.res.y, 1 };
	_optix.pipeline->launch(&launch_interface_buffer,res);

	framebuffer.launch_finish();
}


}
