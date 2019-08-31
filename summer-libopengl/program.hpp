#pragma once


#include "stdafx.hpp"


namespace Summer { namespace OpenGL {


class ShaderBase {
	friend class Program;
	private:
		GLuint _shader;

	protected:
		ShaderBase(GLenum type, char const* source, char const* name);
	public:
		ShaderBase(ShaderBase const&) = delete;
		virtual ~ShaderBase();
};

class ShaderVertex   final : public ShaderBase {
	public:
		explicit ShaderVertex  (char        const* source) : ShaderBase(GL_VERTEX_SHADER,          source, "Vertex"                 ) {}
		explicit ShaderVertex  (std::string const& source) : ShaderVertex  (source.c_str()) {}
		virtual ~ShaderVertex() override = default;
};
class ShaderTessCtrl final : public ShaderBase {
	public:
		explicit ShaderTessCtrl(char        const* source) : ShaderBase(GL_TESS_CONTROL_SHADER,    source, "Tessellation control"   ) {}
		explicit ShaderTessCtrl(std::string const& source) : ShaderTessCtrl(source.c_str()) {}
		virtual ~ShaderTessCtrl() override = default;
};
class ShaderTessEval final : public ShaderBase {
	public:
		explicit ShaderTessEval(char        const* source) : ShaderBase(GL_TESS_EVALUATION_SHADER, source, "Tessellation evaluation") {}
		explicit ShaderTessEval(std::string const& source) : ShaderTessEval(source.c_str()) {}
		virtual ~ShaderTessEval() override = default;
};
class ShaderGeometry final : public ShaderBase {
	public:
		explicit ShaderGeometry(char        const* source) : ShaderBase(GL_GEOMETRY_SHADER,        source, "Geometry"               ) {}
		explicit ShaderGeometry(std::string const& source) : ShaderGeometry(source.c_str()) {}
		virtual ~ShaderGeometry() override = default;
};
class ShaderFragment final : public ShaderBase {
	public:
		explicit ShaderFragment(char        const* source) : ShaderBase(GL_FRAGMENT_SHADER,        source, "Fragment"               ) {}
		explicit ShaderFragment(std::string const& source) : ShaderFragment(source.c_str()) {}
		virtual ~ShaderFragment() override = default;
};


class Program final {
	private:
		GLuint _program;

		std::map<std::string,GLint> mutable _uniform_locs;

		static Program* _program_current;

	private:
		void _attach(ShaderBase const& shader);
		void _link();
	public:
		template <class... TypesShaders> Program(
			TypesShaders const&... shaders
		) {
			_program = glCreateProgram();

			( _attach(shaders), ... );

			_link();
		}
		Program(Program const&) = delete;
		~Program();

		GLint get_loc_uniform  (std::string const& name) const;
		GLint get_loc_attribute(std::string const& name) const;

		void pass_1b  (std::string const& name, bool             value) { glUniform1i(get_loc_uniform(name),value?1:0); }
		void pass_1f  (std::string const& name, float            value) { glUniform1f(get_loc_uniform(name),value); }
		void pass_1i  (std::string const& name, int32_t          value) { glUniform1i(get_loc_uniform(name),value); }
		void pass_1u  (std::string const& name, uint32_t         value) { glUniform1ui(get_loc_uniform(name),value); }
		void pass_3f  (std::string const& name, glm::vec3 const& vec  ) { glUniform3f(get_loc_uniform(name),vec.x,vec.y,vec.z); }
		void pass_4f  (std::string const& name, glm::vec4 const& vec  ) { glUniform4f(get_loc_uniform(name),vec.x,vec.y,vec.z,vec.w); }
		void pass_4x4f(std::string const& name, glm::mat4 const& matr ) { glUniformMatrix4fv(get_loc_uniform(name), 1, GL_FALSE, glm::value_ptr(matr)); }

		static void use(Program* program=nullptr);
};


}}
