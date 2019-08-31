#include "framebuffer.hpp"


namespace Summer { namespace Scene {


#if 0
pixels = new uint8_t[res[1]*res[0]*4];
for (size_t i=0;i<res[0];++i) {
	for (size_t j=0;j<res[1];++j) {
		uint8_t brightness = (i%8>=4)^(j%8>=4) ? 0x00 : 0xFF;
		pixels[4*(j*res[0]+i)  ] = brightness;
		pixels[4*(j*res[0]+i)+1] = brightness;
		pixels[4*(j*res[0]+i)+2] = brightness;
		pixels[4*(j*res[0]+i)+3] = 128;
	}
}
#endif


template<Image2D::FORMAT fmt> Framebuffer::Layer<fmt>::Layer(Vec2zu const& res) :
	texture(res),
	pbo(res,ImageFormatInfo<fmt>::sizeof_texel)
{
	texture.set_gpu_opengl<fmt>();
}

template<Image2D::FORMAT fmt> void Framebuffer::Layer<fmt>::_update_texture() {
	texture.copy_pbo_to_opengl(&pbo);
}

template class Framebuffer::Layer<Image2D::FORMAT::SCALAR_F32  >;
template class Framebuffer::Layer<Image2D::FORMAT::VEC2_F32    >;
template class Framebuffer::Layer<Image2D::FORMAT::VEC3_F32    >;
template class Framebuffer::Layer<Image2D::FORMAT::CIEXYZ_F32  >;
template class Framebuffer::Layer<Image2D::FORMAT::CIEXYZ_A_F32>;
template class Framebuffer::Layer<Image2D::FORMAT::lRGB_F32    >;
template class Framebuffer::Layer<Image2D::FORMAT::lRGB_A_F32  >;
template class Framebuffer::Layer<Image2D::FORMAT::sRGB_U8     >;
template class Framebuffer::Layer<Image2D::FORMAT::sRGB_A_U8   >;


Framebuffer::Layers::Layers(Vec2zu const& res, RenderSettings::LAYERS layers) {
	#define SUMMER_INIT(ENUM,FIELD,FMT)\
		if ((static_cast<uint32_t>(layers)&static_cast<uint32_t>(RenderSettings::LAYERS::ENUM))>0u) {\
			FIELD = new Layer<Image2D::FORMAT::FMT>(res);\
		} else {\
			FIELD = nullptr;\
		}

	sampling_weights_and_count = new Layer<Image2D::FORMAT::VEC2_F32>(res);

	if ((static_cast<uint32_t>(layers)&static_cast<uint32_t>(RenderSettings::LAYERS::MSK_LIGHTING))>0u) {
		lighting_integration      = new Layer<Image2D::FORMAT::CIEXYZ_F32>(res);
		lighting_samples_variance = new Layer<Image2D::FORMAT::SCALAR_F32>(res);
		if ((static_cast<uint32_t>(layers)&static_cast<uint32_t>(RenderSettings::LAYERS::LIGHTING_POSTPROCESSED))>0u) {
			lighting_tmp   = new Layer<Image2D::FORMAT::lRGB_F32>(res);
			lighting_final = new Layer<Image2D::FORMAT::lRGB_F32>(res);
		} else {
			lighting_tmp   = nullptr;
			lighting_final = nullptr;
		}
	} else {
		lighting_integration      = nullptr;
		lighting_samples_variance = nullptr;
		lighting_tmp   = nullptr;
		lighting_final = nullptr;
	}

	SUMMER_INIT(SCENE_COVERAGE,                        scene_coverage,                        SCALAR_F32)
	SUMMER_INIT(SCENE_DISTANCE,                        scene_distance,                        SCALAR_F32)
	SUMMER_INIT(SCENE_DEPTH,                           scene_depth,                           SCALAR_F32)
	SUMMER_INIT(SCENE_NORMALS_GEOMETRIC,               scene_normals_geometric,               VEC3_F32  )
	SUMMER_INIT(SCENE_NORMALS_SHADING,                 scene_normals_shading,                 VEC3_F32  )
	SUMMER_INIT(SCENE_MOTION_VECTORS,                  scene_motion_vectors,                  VEC3_F32  )
	SUMMER_INIT(SCENE_TRIANGLE_BARYCENTRIC_COORDINATES,scene_triangle_barycentric_coordinates,VEC2_F32  )
	SUMMER_INIT(SCENE_TEXTURE_COORDINATES,             scene_texture_coordinates,             VEC2_F32  )
	SUMMER_INIT(SCENE_ALBEDO,                          scene_albedo,                          lRGB_A_F32)
	SUMMER_INIT(SCENE_FRESNEL_TERM,                    scene_fresnel_term,                    VEC2_F32  )

	SUMMER_INIT(STATISTIC_ACCEL_OPS,       stats_accelstruct,    VEC2_F32  )
	SUMMER_INIT(STATISTIC_DEPTH_COMPLEXITY,stats_depthcomplexity,SCALAR_F32)
	SUMMER_INIT(STATISTIC_PHOTON_DENSITY,  stats_photondensity,  SCALAR_F32)

	#undef SUMMER_INIT
}
Framebuffer::Layers::~Layers() {
	delete stats_photondensity;
	delete stats_depthcomplexity;
	delete stats_accelstruct;

	delete scene_fresnel_term;
	delete scene_albedo;
	delete scene_texture_coordinates;
	delete scene_triangle_barycentric_coordinates;
	delete scene_motion_vectors;
	delete scene_normals_shading;
	delete scene_normals_geometric;
	delete scene_depth;
	delete scene_distance;
	delete scene_coverage;

	delete lighting_samples_variance;
	delete lighting_final;
	delete lighting_tmp;
	delete lighting_integration;

	delete sampling_weights_and_count;
}

void Framebuffer::Layers::clear_rendered(CUDA::Context const* context_cuda) {
	sampling_weights_and_count->pbo.set_async(context_cuda,0x00);

	if (lighting_integration!=nullptr) {
		lighting_integration->     pbo.set_async(context_cuda,0x00);
		lighting_samples_variance->pbo.set_async(context_cuda,0x00);
	}

	#define SUMMER_CLEAR(FIELD)\
		if (FIELD!=nullptr) FIELD->pbo.set_async(context_cuda,0x00);

	SUMMER_CLEAR(scene_coverage                        )
	SUMMER_CLEAR(scene_distance                        )
	SUMMER_CLEAR(scene_depth                           )
	SUMMER_CLEAR(scene_normals_geometric               )
	SUMMER_CLEAR(scene_normals_shading                 )
	SUMMER_CLEAR(scene_motion_vectors                  )
	SUMMER_CLEAR(scene_triangle_barycentric_coordinates)
	SUMMER_CLEAR(scene_texture_coordinates             )
	SUMMER_CLEAR(scene_albedo                          )
	SUMMER_CLEAR(scene_fresnel_term                    )

	SUMMER_CLEAR(stats_accelstruct    )
	SUMMER_CLEAR(stats_depthcomplexity)
	SUMMER_CLEAR(stats_photondensity  )

	#undef SUMMER_CLEAR

	assert_cuda(cudaDeviceSynchronize());
}

void Framebuffer::Layers::  _map(CUDA::Context const* context_cuda) {
	sampling_weights_and_count->pbo.map(context_cuda);

	if (lighting_integration!=nullptr) {
		lighting_integration->     pbo.map(context_cuda);
		lighting_samples_variance->pbo.map(context_cuda);
		/*if (lighting_tmp!=nullptr) {
			lighting_tmp->  pbo.map(context_cuda);
			lighting_final->pbo.map(context_cuda);
		}*/
	}

	#define SUMMER_MAP(FIELD)\
		if (FIELD==nullptr); else FIELD->pbo.map(context_cuda);

	SUMMER_MAP(scene_coverage                        )
	SUMMER_MAP(scene_distance                        )
	SUMMER_MAP(scene_depth                           )
	SUMMER_MAP(scene_normals_geometric               )
	SUMMER_MAP(scene_normals_shading                 )
	SUMMER_MAP(scene_motion_vectors                  )
	SUMMER_MAP(scene_triangle_barycentric_coordinates)
	SUMMER_MAP(scene_texture_coordinates             )
	SUMMER_MAP(scene_albedo                          )
	SUMMER_MAP(scene_fresnel_term                    )

	SUMMER_MAP(stats_accelstruct    )
	SUMMER_MAP(stats_depthcomplexity)
	SUMMER_MAP(stats_photondensity  )

	#undef SUMMER_MAP
}
void Framebuffer::Layers::_unmap(                                 ) {
	sampling_weights_and_count->pbo.unmap();

	if (lighting_integration!=nullptr) {
		lighting_integration->     pbo.unmap();
		lighting_samples_variance->pbo.unmap();
		/*if (lighting_tmp!=nullptr) {
			lighting_tmp->  pbo.unmap();
			lighting_final->pbo.unmap();
		}*/
	}

	#define SUMMER_UNMAP(FIELD)\
		if (FIELD==nullptr); else FIELD->pbo.unmap();

	SUMMER_UNMAP(scene_coverage                        )
	SUMMER_UNMAP(scene_distance                        )
	SUMMER_UNMAP(scene_depth                           )
	SUMMER_UNMAP(scene_normals_geometric               )
	SUMMER_UNMAP(scene_normals_shading                 )
	SUMMER_UNMAP(scene_motion_vectors                  )
	SUMMER_UNMAP(scene_triangle_barycentric_coordinates)
	SUMMER_UNMAP(scene_texture_coordinates             )
	SUMMER_UNMAP(scene_albedo                          )
	SUMMER_UNMAP(scene_fresnel_term                    )

	SUMMER_UNMAP(stats_accelstruct    )
	SUMMER_UNMAP(stats_depthcomplexity)
	SUMMER_UNMAP(stats_photondensity  )

	#undef SUMMER_UNMAP
}


static size_t _num_framebuffers = 0;
static OpenGL::Program* _program_draw;

enum DRAW_MODE : int {
	R_IS_BASIC,
	R_IS_HEATMAP,
	G_IS_HEATMAP,
	RG_IS_BASIC,
	RGB_IS_CIEXYZ,
	RGB_IS_lRGB,
	RGB_IS_BASIC
};

Framebuffer::Framebuffer(Vec2zu const& res, RenderSettings::LAYERS layers) :
	res(res), layers(res,layers) DEBUG_ONLY(COMMA _mapped(false))
{
	++_num_framebuffers;
	if (_num_framebuffers==1) {
		_program_draw = new OpenGL::Program(
			OpenGL::ShaderVertex(
				"#version 430 core\n\n"

				"layout(location=0) in vec2 vert_vin;\n\n"

				"out vec2 st_fin;\n\n"

				"void main() {\n"
				"	st_fin = vert_vin*0.5 + vec2(0.5);\n"
				"	gl_Position = vec4( vert_vin,0.0, 1.0 );\n"
				"}\n"
			),
			OpenGL::ShaderFragment(
				"#version 430 core\n\n"

				"uniform int mode;\n"
				"uniform bool reconstruct;\n"
				"layout(binding=0) uniform sampler2D tex2D_weights_and_count;\n"
				"layout(binding=1) uniform sampler2D tex2D_visualize;\n\n"

				"in vec2 st_fin;\n\n"

				"layout(location=0) out vec4 color;\n\n"

				//TODO: real heatmap!
				"vec3 heatmap(float intensity) {\n"
				"	return vec3( intensity/10.0, 0.0, 0.0 );\n"
				"}\n"
				//TODO: real conversion!
				"vec3 ciexyz_to_lrgb(vec3 ciexyz) {\n"
				"	return ciexyz;\n"
				"}\n\n"

				"void main() {\n"
				"	vec4 tap = texture(tex2D_visualize,st_fin);\n"
				"	if (reconstruct) {\n"
				"		tap /= texture( tex2D_weights_and_count, st_fin ).r;\n"
				"	}\n\n"

				"	vec4 result;\n"
				"	switch (mode) {\n"
				"		case "+std::to_string(DRAW_MODE::R_IS_BASIC   )+":\n"
				"			result=vec4( vec3(tap.r), 1.0 ); break;\n"
				"		case "+std::to_string(DRAW_MODE::R_IS_HEATMAP )+":\n"
				"			result=vec4( heatmap(tap.r), 1.0 ); break;\n"
				"		case "+std::to_string(DRAW_MODE::G_IS_HEATMAP )+":\n"
				"			result=vec4( heatmap(tap.g), 1.0 ); break;\n"
				"		case "+std::to_string(DRAW_MODE::RG_IS_BASIC  )+":\n"
				"			result=vec4( tap.rg,0.0, 1.0 ); break;\n"
				"		case "+std::to_string(DRAW_MODE::RGB_IS_CIEXYZ)+":\n"
				"			result=vec4( ciexyz_to_lrgb(tap.rgb), 1.0 ); break;\n"
				"		case "+std::to_string(DRAW_MODE::RGB_IS_lRGB  )+":\n" //fallthrough
				"		case "+std::to_string(DRAW_MODE::RGB_IS_BASIC )+":\n"
				"			result=vec4( tap.rgb, 1.0 ); break;\n"
				"		default:\n"
				"			result=vec4( 1.0,0.0,1.0, 1.0 ); break;\n"
				"	}\n\n"

				"	color = result;\n"
				"}\n"
			)
		);
	}
}
Framebuffer::~Framebuffer() {
	assert_term(_num_framebuffers>0,"Implementation error!");
	--_num_framebuffers;
	if (_num_framebuffers==0) {
		delete _program_draw;
	}
}

Framebuffer::InterfaceGPU Framebuffer::get_interface() const {
	assert_term(_mapped,"Must map buffers by calling `.launch_prepare(...)` first!");
	return {
		res,
		{
			layers.sampling_weights_and_count->pbo.mapped_ptr,

			#define SUMMER_PTR(FIELD)\
				layers.FIELD==nullptr ? nullptr : layers.FIELD->pbo.mapped_ptr

			SUMMER_PTR(lighting_integration     ),
			SUMMER_PTR(lighting_samples_variance),

			SUMMER_PTR(scene_coverage                        ),
			SUMMER_PTR(scene_distance                        ),
			SUMMER_PTR(scene_depth                           ),
			SUMMER_PTR(scene_normals_geometric               ),
			SUMMER_PTR(scene_normals_shading                 ),
			SUMMER_PTR(scene_motion_vectors                  ),
			SUMMER_PTR(scene_triangle_barycentric_coordinates),
			SUMMER_PTR(scene_texture_coordinates             ),
			SUMMER_PTR(scene_albedo                          ),
			SUMMER_PTR(scene_fresnel_term                    ),

			SUMMER_PTR(stats_accelstruct    ),
			SUMMER_PTR(stats_depthcomplexity),
			SUMMER_PTR(stats_photondensity  )

			#undef SUMMER_PTR
		}
	};
}

void Framebuffer::launch_prepare(CUDA::Context const* context_cuda) {
	assert_term(!_mapped,"Already mapped!");
	layers.  _map(context_cuda);
	DEBUG_ONLY(_mapped = true;)
}
void Framebuffer::launch_finish (                                 ) {
	assert_term(_mapped,"Not mapped!");
	layers._unmap();
	DEBUG_ONLY(_mapped = false;)
}

void Framebuffer::process_and_draw(RenderSettings const& render_settings) {
	GLuint handle;
	DRAW_MODE draw_mode;
	layers.sampling_weights_and_count->_update_texture();
	switch (render_settings.layer_primary_output) {
		case RenderSettings::LAYERS::SAMPLING_WEIGHTS:
			handle = layers.sampling_weights_and_count->texture.data.gpu_gl.handle;
			draw_mode = DRAW_MODE::R_IS_HEATMAP;
			break;
		case RenderSettings::LAYERS::SAMPLING_COUNT:
			handle = layers.sampling_weights_and_count->texture.data.gpu_gl.handle;
			draw_mode = DRAW_MODE::G_IS_HEATMAP;
			break;

		#define SUMMER_CASE(ENUMLAYERS,FIELD,ENUMDRAW)\
			case RenderSettings::LAYERS::ENUMLAYERS:\
				layers.FIELD->_update_texture();\
				handle = layers.FIELD->texture.data.gpu_gl.handle;\
				draw_mode = DRAW_MODE::ENUMDRAW;\
				break;

		SUMMER_CASE(LIGHTING_RAW,          lighting_integration,     RGB_IS_CIEXYZ)
		SUMMER_CASE(LIGHTING_POSTPROCESSED,lighting_final,           RGB_IS_lRGB  )
		SUMMER_CASE(LIGHTING_VARIANCE,     lighting_samples_variance,R_IS_HEATMAP )

		SUMMER_CASE(SCENE_COVERAGE,                        scene_coverage,                        R_IS_BASIC  )
		SUMMER_CASE(SCENE_DISTANCE,                        scene_distance,                        R_IS_BASIC  )
		SUMMER_CASE(SCENE_DEPTH,                           scene_depth,                           R_IS_BASIC  )
		SUMMER_CASE(SCENE_NORMALS_GEOMETRIC,               scene_normals_geometric,               RGB_IS_BASIC)
		SUMMER_CASE(SCENE_NORMALS_SHADING,                 scene_normals_shading,                 RGB_IS_BASIC)
		SUMMER_CASE(SCENE_MOTION_VECTORS,                  scene_motion_vectors,                  RGB_IS_BASIC)
		SUMMER_CASE(SCENE_TRIANGLE_BARYCENTRIC_COORDINATES,scene_triangle_barycentric_coordinates,RG_IS_BASIC )
		SUMMER_CASE(SCENE_TEXTURE_COORDINATES,             scene_texture_coordinates,             RG_IS_BASIC )
		SUMMER_CASE(SCENE_ALBEDO,                          scene_albedo,                          RGB_IS_lRGB )
		SUMMER_CASE(SCENE_FRESNEL_TERM,                    scene_fresnel_term,                    RG_IS_BASIC )

		SUMMER_CASE(STATISTIC_ACCEL_OPS,       stats_accelstruct,    RG_IS_BASIC )
		SUMMER_CASE(STATISTIC_DEPTH_COMPLEXITY,stats_depthcomplexity,R_IS_HEATMAP)
		SUMMER_CASE(STATISTIC_PHOTON_DENSITY,  stats_photondensity,  R_IS_HEATMAP)

		#undef SUMMER_CASE

		nodefault;
	}
	bool reconstruct = handle!=layers.sampling_weights_and_count->texture.data.gpu_gl.handle;
	bool sRGB;
	switch (draw_mode) {
		case DRAW_MODE::R_IS_BASIC:  [[fallthrough]];
		case DRAW_MODE::RG_IS_BASIC: [[fallthrough]];
		case DRAW_MODE::RGB_IS_BASIC:
			sRGB = false;
			break;
		default:
			sRGB = true;
			break;
	}

	glViewport(0,0,static_cast<GLsizei>(res[0]),static_cast<GLsizei>(res[1]));

	glClear(GL_COLOR_BUFFER_BIT);

	if (sRGB) glEnable(GL_FRAMEBUFFER_SRGB);

	OpenGL::Program::use(_program_draw);
	_program_draw->pass_1i("mode",       draw_mode  );
	_program_draw->pass_1i("reconstruct",reconstruct);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D,layers.sampling_weights_and_count->texture.data.gpu_gl.handle);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D,handle);

	#if 1
	glBegin(GL_QUADS);
	glVertex2f(-1.0f,-1.0f);
	glVertex2f( 1.0f,-1.0f);
	glVertex2f( 1.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glEnd();
	#endif

	if (sRGB) glDisable(GL_FRAMEBUFFER_SRGB);

	/*CUDA::BufferCPUManaged pixels_cpu( layers.rgba.pbo.size );
	layers.rgba.pbo.copy_to_buffer(context_cuda,&pixels_cpu);

	glClear(GL_COLOR_BUFFER_BIT);

	glDrawPixels(static_cast<GLsizei>(res[0]),static_cast<GLsizei>(res[1]),GL_RGBA,GL_UNSIGNED_BYTE,pixels_cpu.ptr);*/
}


}}
