#include "program.hpp"


namespace Summer { namespace OpenGL {


static std::string _get_globject_infolog(GLuint id) {
	int max_length = -1;

	bool is_shader = glIsShader(id)==GL_TRUE;

	(is_shader?glGetShaderiv:glGetProgramiv)(id, GL_INFO_LOG_LENGTH, &max_length);

	//Result should be larger than zero, if only for the null byte, but evidently sometimes it
	//	produces zero, so handle that case specially.
	if        (max_length==0) {
		return "";
	} else if (max_length >0) {
		char* temp_output = new char[static_cast<size_t>(max_length)];

		(is_shader?glGetShaderInfoLog:glGetProgramInfoLog)(id, static_cast<GLsizei>(max_length),nullptr, temp_output);

		std::string output = temp_output;
		delete [] temp_output;

		return output;
	} else {
		puts(
			"Invalid result of shading API log get!  Since there was no error reported and a context is set, "
			"this indicates a driver bug.  Assuming log was empty."
		);

		return "";
	}
}


ShaderBase::ShaderBase(GLenum type, char const* source, char const* name) {
	_shader = glCreateShader(type);

	glShaderSource(_shader, 1,&source, nullptr);

	glCompileShader(_shader);

	std::string log = _get_globject_infolog(_shader);
	if (log.empty()) {
		//printf("%s compilation output:",name);
		//printf(" (none)\n");
	} else {
		printf("%s compilation output:",name);
		printf("\n%s\n",log.c_str());

		printf("  1: ");
		unsigned linenum = 1;
		while (*source!='\0') {
			if (*source=='\n') {
				printf("\n%3u: ",++linenum);
			} else {
				putc(*source,stdout);
			}
			++source;
		}
		printf("\n");

		printf("ENTER to continue."); getchar();
	}
}
ShaderBase::~ShaderBase() {
	glDeleteShader(_shader);
}


void Program::_attach(ShaderBase const& shader) {
	glAttachShader(_program,shader._shader);
}
void Program::_link() {
	glLinkProgram(_program);

	std::string log = _get_globject_infolog(_program);
	if (log.empty()) {
		//printf("Program link output:\n%s\n",log.c_str());
	} else {
		printf("Program link output:\n%s\n",log.c_str());

		printf("ENTER to continue."); getchar();
	}
}

Program::~Program() {
	if (_program_current==this) Program::use(nullptr);

	glDeleteProgram(_program);
}

GLint Program::get_loc_uniform  (std::string const& name) const {
	auto iter = _uniform_locs.find(name);
	if (iter!=_uniform_locs.cend()) {
		return iter->second;
	} else {
		GLint result;
		if (Program::_program_current==this) {
			result = glGetUniformLocation(_program,name.c_str());
		} else {
			glUseProgram(_program);
			result = glGetUniformLocation(_program,name.c_str());
			glUseProgram(Program::_program_current!=nullptr?Program::_program_current->_program:0);
		}
		_uniform_locs[name] = result;
		return result;
	}
}
GLint Program::get_loc_attribute(std::string const& name) const {
	if (Program::_program_current==this) {
		return glGetAttribLocation(_program,name.c_str());
	} else {
		glUseProgram(_program);
		GLint result = glGetAttribLocation(_program,name.c_str());
		glUseProgram(Program::_program_current!=nullptr?Program::_program_current->_program:0);
		return result;
	}
}

void Program::use(Program* program/*=nullptr*/) {
	if (_program_current==program);
	else {
		if (program!=nullptr) glUseProgram(program->_program);
		else                  glUseProgram(0                );
		_program_current = program;
	}
}

Program* Program::_program_current = nullptr;


}}
