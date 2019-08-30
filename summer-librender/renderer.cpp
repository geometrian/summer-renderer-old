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

		/*Scene::Scene const* scene = scenegraph->scenes[0];
		for (Scene::Node const* root_node : scene->root_nodes) {
			std::function<void(Scene::Node const*)> add_node = [&](Scene::Node const* node) -> void {
				for (Scene::Node const* child : node->children) {
					add_node(child);
				}
				for (Scene::Object const* object : node->objects) {
					for (Scene::Object::Mesh const* mesh : object->meshes) {
						for (OptiX::ProgramsHitOps const* programs_hitops : programsets_hitops) {
							sbt_builder.hitsops.emplace_back(std::make_pair( programs_hitops, new DataSBT_HitOps(mesh) ));
						}
					}
				}
			};
			add_node(root_node);
		}*/
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
		_optix.program_sets["raygen-forward"         ] = new OptiX::ProgramRaygen ( _optix.context, _optix.module,"__raygen__forward"       );

		//Miss
		_optix.program_sets["miss-color"             ] = new OptiX::ProgramMiss   ( _optix.context, _optix.module,"__miss__color"           );
		_optix.program_sets["miss-pathtrace"         ] = new OptiX::ProgramMiss   ( _optix.context, _optix.module,"__miss__pathtrace"       );
		_optix.program_sets["miss-pathtrace-shadow"  ] = new OptiX::ProgramMiss   ( _optix.context, _optix.module,"__miss__pathtrace_shadow");

		//Hit operations
		_optix.program_sets["hitops-albedo"          ] = new OptiX::ProgramsHitOps(
			_optix.context,
			_optix.module, "__closesthit__albedo",
			nullptr,       nullptr,
			nullptr,       nullptr
		);
		_optix.program_sets["hitops-normals"         ] = new OptiX::ProgramsHitOps(
			_optix.context,
			_optix.module, "__closesthit__normals",
			nullptr,       nullptr,
			nullptr,       nullptr
		);
		_optix.program_sets["hitops-pathtrace"       ] = new OptiX::ProgramsHitOps(
			_optix.context,
			_optix.module, "__closesthit__pathtrace",
			nullptr,       nullptr, //_optix.module, "__anyhit__pathtrace",
			nullptr,       nullptr
		);
		_optix.program_sets["hitops-pathtrace-shadow"] = new OptiX::ProgramsHitOps(
			_optix.context,
			nullptr,       nullptr,
			_optix.module, "__anyhit__pathtrace_shadow",
			nullptr,       nullptr
		);
		_optix.program_sets["hitops-texcs"           ] = new OptiX::ProgramsHitOps(
			_optix.context,
			_optix.module, "__closesthit__texcs",
			nullptr,       nullptr,
			nullptr,       nullptr
		);
		_optix.program_sets["hitops-tri-bary"        ] = new OptiX::ProgramsHitOps(
			_optix.context,
			_optix.module, "__closesthit__tri_bary",
			nullptr,       nullptr,
			nullptr,       nullptr
		);
	}

	//Set up integrators
	{
		integrators["albedo"   ] = new Integrator(
			this, scenegraph,
			  static_cast<OptiX::ProgramRaygen* >(_optix.program_sets.at("raygen-forward"  )),
			{ static_cast<OptiX::ProgramMiss*   >(_optix.program_sets.at("miss-color"      )) },
			{ static_cast<OptiX::ProgramsHitOps*>(_optix.program_sets.at("hitops-albedo"   )) }
		);
		integrators["normals"  ] = new Integrator(
			this, scenegraph,
			  static_cast<OptiX::ProgramRaygen* >(_optix.program_sets.at("raygen-forward"  )),
			{ static_cast<OptiX::ProgramMiss*   >(_optix.program_sets.at("miss-color"      )) },
			{ static_cast<OptiX::ProgramsHitOps*>(_optix.program_sets.at("hitops-normals"  )) }
		);
		integrators["pathtrace"] = new Integrator(
			this, scenegraph,
			static_cast<OptiX::ProgramRaygen* >(_optix.program_sets.at("raygen-forward"  )),
			{
				static_cast<OptiX::ProgramMiss*   >(_optix.program_sets.at("miss-pathtrace"         )),
				static_cast<OptiX::ProgramMiss*   >(_optix.program_sets.at("miss-pathtrace-shadow"  ))
			},
			{
				static_cast<OptiX::ProgramsHitOps*>(_optix.program_sets.at("hitops-pathtrace"       )),
				static_cast<OptiX::ProgramsHitOps*>(_optix.program_sets.at("hitops-pathtrace-shadow"))
			}
		);
		integrators["texcs"    ] = new Integrator(
			this, scenegraph,
			  static_cast<OptiX::ProgramRaygen* >(_optix.program_sets.at("raygen-forward"  )),
			{ static_cast<OptiX::ProgramMiss*   >(_optix.program_sets.at("miss-color"      )) },
			{ static_cast<OptiX::ProgramsHitOps*>(_optix.program_sets.at("hitops-texcs"    )) }
		);
		integrators["tri-bary" ] = new Integrator(
			this, scenegraph,
			  static_cast<OptiX::ProgramRaygen* >(_optix.program_sets.at("raygen-forward"  )),
			{ static_cast<OptiX::ProgramMiss*   >(_optix.program_sets.at("miss-color"      )) },
			{ static_cast<OptiX::ProgramsHitOps*>(_optix.program_sets.at("hitops-tri-bary" )) }
		);
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

void Renderer::render(size_t scene_index, size_t camera_index, float timestamp, std::string const& integrator_name) const {
	assert_term(scene_index==0,"Not implemented!"); //SBT build above

	Scene::Scene const* scene = scenegraph->scenes[scene_index];
	Scene::Framebuffer& framebuffer = scene->cameras[camera_index]->framebuffer;

	framebuffer.launch_prepare(_cuda.context);

	Scene::Scene::InterfaceGPU const& interface_gpu = scene->get_interface(camera_index);
	integrators.at(integrator_name)->render(interface_gpu);

	framebuffer.launch_finish();
}


}
