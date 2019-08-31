#include "stdafx.hpp"


static Vec2zu const res = Vec2zu(1024,768);
//static Vec2zu const res = Vec2zu(2560,1440);


#define SCENE_NUMBER 3
#if   SCENE_NUMBER == -1
	#define SCENE_NAME "AntiqueCamera"
	static float camera_angles[2] = { 30.0f, 20.0f };
	static float camera_radius = 10.0f;
	static Vec3f camera_center( 0.0f, 6.0f, 0.0f );
#elif SCENE_NUMBER == 0
	#define SCENE_NAME "BarramundiFish"
	static float camera_angles[2] = { 150.0f, 10.0f };
	static float camera_radius = 0.6f;
	static Vec3f camera_center( 0.0f, 0.1f, 0.0f );
#elif SCENE_NUMBER == 1
	//#define SCENE_NAME "BoomBox"
	#define SCENE_NAME "BoomBoxWithAxes"
	static float camera_angles[2] = { 30.0f, 20.0f };
	static float camera_radius = 0.05f;
	static Vec3f camera_center( 0.0f, 0.01f, 0.0f );
#elif SCENE_NUMBER == 2
	//#define SCENE_NAME "Box"
	//#define SCENE_NAME "BoxInterleaved"
	#define SCENE_NAME "BoxTextured"
	static float camera_angles[2] = { 60.0f, 22.5f };
	static float camera_radius = 2.0f;
	static Vec3f camera_center( 0.0f, 0.0f, 0.0f );
#elif SCENE_NUMBER == 3
	#define SCENE_NAME "Buggy"
	static float camera_angles[2] = { 140.0f, 22.5f };
	static float camera_radius = 150.0f;
	static Vec3f camera_center( 20.0f, 20.0f, 20.0f );
#elif SCENE_NUMBER == 4
	#define SCENE_NAME "DamagedHelmet"
	static float camera_angles[2] = { 60.0f, 22.5f };
	static float camera_radius = 5.0f;
	static Vec3f camera_center( 0.0f, 0.0f, 0.0f );
#elif SCENE_NUMBER == 5
	#define SCENE_NAME "Duck"
	static float camera_angles[2] = { 60.0f, 22.5f };
	static float camera_radius = 4.0f;
	static Vec3f camera_center( 0.0f, 0.7f, 0.0f );
#elif SCENE_NUMBER == 6
	#define SCENE_NAME "MetalRoughSpheres"
	static float camera_angles[2] = { 60.0f, 22.5f };
	static float camera_radius = 5.0f;
	static Vec3f camera_center( 0.0f, 0.0f, 0.0f );
#elif SCENE_NUMBER == 7
	#define SCENE_NAME "ReciprocatingSaw"
	static float camera_angles[2] = { 60.0f, 22.5f };
	static float camera_radius = 5.0f;
	static Vec3f camera_center( 0.0f, 0.0f, 0.0f );
#elif SCENE_NUMBER == 8
	#define SCENE_NAME "SimpleMeshes"
	static float camera_angles[2] = { 60.0f, 22.5f };
	static float camera_radius = 5.0f;
	static Vec3f camera_center( 0.0f, 0.0f, 0.0f );
#elif SCENE_NUMBER == 9
	#define SCENE_NAME "Sponza"
	static float camera_angles[2] = { 60.0f, 22.5f };
	static float camera_radius = 1.0f;
	static Vec3f camera_center( 0.0f, 4.0f, 0.0f );
#endif

static Summer::RenderSettings* render_settings;

static bool render_dirty = true;


#ifdef BUILD_DEBUG
inline static void _callback_err_glfw(int /*error*/, char const* description) {
	fprintf(stderr, "Error: %s\n", description);
}
#endif

inline static void _callback_key(GLFWwindow* window, int key,int /*scancode*/, int action, int mods) {
	if (action==GLFW_PRESS) {
		bool alt_pressed =
			glfwGetKey(window,GLFW_KEY_LEFT_ALT )==GLFW_PRESS ||
			glfwGetKey(window,GLFW_KEY_RIGHT_ALT)==GLFW_PRESS
		;

		switch (key) {
			case GLFW_KEY_ESCAPE:
				glfwSetWindowShouldClose(window, GLFW_TRUE);
				break;
			case GLFW_KEY_A:
				if (alt_pressed) render_settings->layer_primary_output=Summer::RenderSettings::LAYERS::SCENE_ALBEDO;
				break;
			case GLFW_KEY_B:
				if (alt_pressed) render_settings->layer_primary_output=Summer::RenderSettings::LAYERS::SCENE_TRIANGLE_BARYCENTRIC_COORDINATES;
				break;
			case GLFW_KEY_C:
				if (alt_pressed) render_settings->layer_primary_output=Summer::RenderSettings::LAYERS::SAMPLING_COUNT;
				break;
			case GLFW_KEY_N:
				if (alt_pressed) render_settings->layer_primary_output=Summer::RenderSettings::LAYERS::SCENE_NORMALS_SHADING;
				break;
			case GLFW_KEY_P:
				if (alt_pressed) render_settings->layer_primary_output=Summer::RenderSettings::LAYERS::LIGHTING_RAW;
				break;
			case GLFW_KEY_T:
				if (alt_pressed) render_settings->layer_primary_output=Summer::RenderSettings::LAYERS::SCENE_TEXTURE_COORDINATES;
				break;
			default:
				break;
		}
	}
}

static double _last_mpos[2] = {0.0,0.0};
static void _callback_mpos(GLFWwindow* window, double xpos,double ypos) {
	if (glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS) {
		camera_angles[0] += static_cast<float>( xpos - _last_mpos[0] );
		camera_angles[1] += static_cast<float>( ypos - _last_mpos[1] );
		render_dirty = true;
	}
	_last_mpos[0] = xpos;
	_last_mpos[1] = ypos;
}
static void _callback_scroll(GLFWwindow* window, double xoffset,double yoffset) {
	if (yoffset>0.0) camera_radius*=0.9f;
	else             camera_radius/=0.9f;
	render_dirty = true;
}


int main(int /*argc*/, char* /*argv*/[]) {
	#if defined _WIN32 && defined BUILD_DEBUG
		_CrtSetDbgFlag(0xFFFFFFFF);
	#endif

	Summer::RenderSettings backing;
	render_settings = &backing;

	{
		GLFWwindow* window;
		{
			#ifdef BUILD_DEBUG
				glfwSetErrorCallback(_callback_err_glfw);
			#endif

			glfwInit();

			glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
			glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
			#ifdef BUILD_DEBUG
				glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
			#endif

			glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
			#ifdef BUILD_COMPILER_MSVC
				glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
			#endif

			window = glfwCreateWindow(
				static_cast<int>(res[0]), static_cast<int>(res[1]),
				"Summer Renderer",
				nullptr,
				nullptr
			);

			glfwSetKeyCallback      (window, _callback_key   );
			glfwSetCursorPosCallback(window, _callback_mpos  );
			glfwSetScrollCallback   (window, _callback_scroll);

			glfwMakeContextCurrent(window);
			gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
			#ifdef BUILD_DEBUG
				//glDebugMessageCallback(_callback_err_gl,nullptr);
			#endif

			//glfwSwapInterval(0);

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		}

		Summer::Scene::SceneGraph* scenegraph =
			//Summer::Scene::load_new_gltf("../../../../../prebuilt-data/objects/glTF-Sample-Models/2.0/" SCENE_NAME "/glTF-Binary/" SCENE_NAME ".glb")
			Summer::Scene::load_new_gltf("../../../../../prebuilt-data/objects/glTF-Sample-Models/2.0/" SCENE_NAME "/glTF/" SCENE_NAME ".gltf")
		;
		scenegraph->cameras.emplace_back(new Summer::Scene::Camera(
			Summer::Scene::Camera::TYPE::LOOKAT, res,static_cast<Summer::RenderSettings::LAYERS>(~0u)
		));
		scenegraph->scenes[0]->cameras.emplace_back(scenegraph->cameras.back());

		Summer::Renderer renderer(scenegraph);

		render_settings->lighting_integrator  = Summer::RenderSettings::LIGHTING_INTEGRATOR::PATH_TRACING;
		render_settings->layer_primary_output = Summer::RenderSettings::LAYERS::LIGHTING_RAW;
		render_settings->index_scene = 0;
		render_settings->index_camera = 0;
		render_settings->time_range[0] = 0.0f;
		render_settings->time_range[1] = 1.0f;

		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();

			if (render_dirty) {
				renderer.reset();
			}

			Summer::Scene::Camera* camera = scenegraph->cameras.back();
			camera->lookat.position = camera_center + Vec3f(
				camera_radius * std::cos(glm::radians(camera_angles[0])) * std::cos(glm::radians(camera_angles[1])),
				camera_radius                                            * std::sin(glm::radians(camera_angles[1])),
				camera_radius * std::sin(glm::radians(camera_angles[0])) * std::cos(glm::radians(camera_angles[1]))
			);
			camera->lookat.center = camera_center;
			camera->lookat.up = Vec3f(0,1,0);

			renderer.render(*render_settings);

			render_dirty = false;

			camera->framebuffer.process_and_draw(*render_settings);

			glfwSwapBuffers(window);
		}

		delete scenegraph;

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	#if defined _WIN32 && defined BUILD_DEBUG
		if (_CrtDumpMemoryLeaks()) {
			fprintf(stderr,"Memory leaks detected!\n");
			//printf("Press ENTER to exit.\n"); getchar();
		}
	#endif

	return 0;
}
