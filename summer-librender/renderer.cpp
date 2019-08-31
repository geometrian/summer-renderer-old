#include "renderer.hpp"

#include "kernels/sbt-entries.hpp"

#include "scene/framebuffer.hpp"
#include "scene/scenegraph.hpp"


extern "C" unsigned char const ptx_embed___summer_librender___kernels___compile_all_kernels_cu[];


namespace Summer {


Renderer::Integrator::Integrator(
	Renderer* parent, Scene::SceneGraph* scenegraph,
	            OptiX::ProgramRaygen*          program_raygen,
	std::vector<OptiX::ProgramMiss*   > const& programs_miss,
	std::vector<OptiX::ProgramsHitOps*> const& programsets_hitops
) {
	OptiX::ShaderBindingTable::Builder<DataSBT_Raygen,DataSBT_Miss,DataSBT_HitOps> sbt_builder;
	{
		sbt_builder.raygen = std::make_pair( program_raygen, new DataSBT_Raygen );

		for (OptiX::ProgramMiss const* program_miss : programs_miss) {
			sbt_builder.miss.emplace_back(std::make_pair( program_miss, new DataSBT_Miss ));
		}

		for (Scene::Object const* object : scenegraph->objects) {
			for (Scene::Object::Mesh const* mesh : object->meshes) {
				assert_term(programsets_hitops.size()<=SUMMER_MAX_RAYTYPES,"Too many ray types!");
				for (OptiX::ProgramsHitOps const* programs_hitops : programsets_hitops) {
					sbt_builder.hitsops.emplace_back(std::make_pair( programs_hitops, new DataSBT_HitOps(mesh) ));
				}
				for (size_t i=programsets_hitops.size();i<SUMMER_MAX_RAYTYPES;++i) {
					sbt_builder.hitsops.emplace_back(std::make_pair( nullptr, nullptr ));
				}
			}
		}
	}
	sbt = new OptiX::ShaderBindingTable(sbt_builder);
	                                             delete sbt_builder.raygen.second;
	for (auto const& iter : sbt_builder.miss   ) delete iter.              second;
	for (auto const& iter : sbt_builder.hitsops) delete iter.              second;

	pipeline = new OptiX::Pipeline( parent->_optix.context, parent->_optix.pipeline_opts, sbt );
}
Renderer::Integrator::~Integrator() {
	delete pipeline;

	delete sbt;
}

void Renderer::Integrator::render(Scene::Scene::InterfaceGPU const& interface_gpu) const {
	//TODO: move outside
	CUDA::BufferGPUManaged launch_interface_buffer( sizeof(Scene::Scene::InterfaceGPU),&interface_gpu );
	size_t res[3] = { interface_gpu.camera.framebuffer.res.x, interface_gpu.camera.framebuffer.res.y, 1 };
	pipeline->launch(&launch_interface_buffer,res);
}


Renderer::Renderer(Scene::SceneGraph* scenegraph) : scenegraph(scenegraph) {
	//CUDA setup
	{
		_cuda.device  = new CUDA::Device(0);
		_cuda.context = new CUDA::Context(_cuda.device);
	}

	//OptiX setup
	{
		_optix.context  = new OptiX::Context(_cuda.context);

		#define SUMMER_QUERY(PROPERTY,FIELD)\
			assert_optix(optixDeviceContextGetProperty( _optix.context->context, OptixDeviceProperty::PROPERTY, &_optix.properties.FIELD,sizeof(uint32_t) ))
		SUMMER_QUERY(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH,                  max_trace_depth        );
		SUMMER_QUERY(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,      max_graph_depth        );
		SUMMER_QUERY(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,           max_prim_in_accelstruct);
		SUMMER_QUERY(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,            max_inst_in_accelstruct);
		SUMMER_QUERY(OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,                         rtcore_version         );
		SUMMER_QUERY(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,                  max_instanceid         );
		SUMMER_QUERY(OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK,numbits_vis_msk        );
		SUMMER_QUERY(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS,          max_sbt_in_accelstruct ); //TODO: documentation confusing/wrong?
		SUMMER_QUERY(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET,                   max_sbt_offset         );
		#undef SUMMER_QUERY
	}

	//Upload scenegraph to GPU
	scenegraph->upload(_optix.context);

	//Load shaders
	{
		//PTX Module
		_optix.module = new OptiX::CompiledModule(_optix.context,_optix.pipeline_opts,reinterpret_cast<char const*>(ptx_embed___summer_librender___kernels___compile_all_kernels_cu));

		//Ray generation
		_optix.program_sets["raygen-lightnone"  ] = new OptiX::ProgramRaygen( _optix.context, _optix.module,"__raygen__lightnone"   );
		_optix.program_sets["raygen-pathtracing"] = new OptiX::ProgramRaygen( _optix.context, _optix.module,"__raygen__pathtracing" );

		//Miss
		_optix.program_sets["miss-lightnone"         ] = new OptiX::ProgramMiss( _optix.context, _optix.module,"__miss__lightnone"          );
		_optix.program_sets["miss-pathtracing-normal"] = new OptiX::ProgramMiss( _optix.context, _optix.module,"__miss__pathtracing_normal" );
		_optix.program_sets["miss-pathtracing-shadow"] = new OptiX::ProgramMiss( _optix.context, _optix.module,"__miss__pathtracing_shadow" );

		//Hit operations
		_optix.program_sets["hitops-lightnone"         ] = new OptiX::ProgramsHitOps(_optix.context,
			_optix.module, "__closesthit__lightnone",
			_optix.module, "__anyhit__lightnone",
			nullptr,       nullptr
		);
		_optix.program_sets["hitops-pathtracing-normal"] = new OptiX::ProgramsHitOps(_optix.context,
			_optix.module, "__closesthit__pathtracing_normal",
			_optix.module, "__anyhit__pathtracing_normal",
			nullptr,       nullptr
		);
		_optix.program_sets["hitops-pathtracing-shadow"] = new OptiX::ProgramsHitOps(_optix.context,
			nullptr,       nullptr,
			_optix.module, "__anyhit__pathtracing_shadow",
			nullptr,       nullptr
		);
	}

	//Set up integrators
	{
		integrators[RenderSettings::LIGHTING_INTEGRATOR::NONE] = new Integrator(
			this, scenegraph,
			  static_cast<OptiX::ProgramRaygen* >(_optix.program_sets.at("raygen-lightnone")),
			{ static_cast<OptiX::ProgramMiss*   >(_optix.program_sets.at("miss-lightnone"  )) },
			{ static_cast<OptiX::ProgramsHitOps*>(_optix.program_sets.at("hitops-lightnone")) }
		);

		//RenderSettings::LIGHTING_INTEGRATOR::AMBIENT_OCCLUSION
		//RenderSettings::LIGHTING_INTEGRATOR::DIRECT_LIGHTING_UNSHADOWED
		//RenderSettings::LIGHTING_INTEGRATOR::SHADOWS

		//RenderSettings::LIGHTING_INTEGRATOR::DIRECT_LIGHTING
		//RenderSettings::LIGHTING_INTEGRATOR::WHITTED
		//RenderSettings::LIGHTING_INTEGRATOR::COOK

		integrators[RenderSettings::LIGHTING_INTEGRATOR::PATH_TRACING] = new Integrator(
			this, scenegraph,
			  static_cast<OptiX::ProgramRaygen* >(_optix.program_sets.at("raygen-pathtracing")),
			{
				static_cast<OptiX::ProgramMiss*   >(_optix.program_sets.at("miss-pathtracing-normal")),
				static_cast<OptiX::ProgramMiss*   >(_optix.program_sets.at("miss-pathtracing-shadow"))
			},
			{
				static_cast<OptiX::ProgramsHitOps*>(_optix.program_sets.at("hitops-pathtracing-normal")),
				static_cast<OptiX::ProgramsHitOps*>(_optix.program_sets.at("hitops-pathtracing-shadow"))
			}
		);
		//RenderSettings::LIGHTING_INTEGRATOR::LIGHT_TRACING
		//RenderSettings::LIGHTING_INTEGRATOR::BIDIRECTIONAL_PATH_TRACING
		//RenderSettings::LIGHTING_INTEGRATOR::METROPOLIS_LIGHT_TRANSPORT

		//RenderSettings::LIGHTING_INTEGRATOR::PHOTONMAPPING

		//RenderSettings::LIGHTING_INTEGRATOR::VERTEX_CONNECTION_MERGING
	}
}
Renderer::~Renderer() {
	{
		for (auto const& iter : integrators) {
			delete iter.second;
		}

		for (auto const& iter : _optix.program_sets) {
			delete iter.second;
		}

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

void Renderer::reset() {
	for (Scene::Camera* camera : scenegraph->cameras) {
		camera->framebuffer.layers.clear_rendered(_cuda.context);
	}
}

void Renderer::render(RenderSettings const& render_settings) const {
	assert_term(render_settings.index_scene==0,"Not implemented!"); //SBT build above

	Scene::Scene const* scene = scenegraph->scenes[render_settings.index_scene];
	Scene::Framebuffer& framebuffer = scene->cameras[render_settings.index_camera]->framebuffer;

	framebuffer.launch_prepare(_cuda.context);

	Scene::Scene::InterfaceGPU const& interface_gpu = scene->get_interface(render_settings.index_camera);
	integrators.at(render_settings.lighting_integrator)->render(interface_gpu);

	framebuffer.launch_finish();
}


}
