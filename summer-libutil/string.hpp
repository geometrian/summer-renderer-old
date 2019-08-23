#pragma once


#include "stdafx.hpp"


#ifndef BUILD_COMPILER_NVCC


#include <cstdint>
#include <deque>
#include <list>
#include <string>


namespace std {


string  to_string (wstring const& str);
wstring to_wstring( string const& str);

static_assert(sizeof(char)==1,"Implementation error!");
inline string to_string(uint8_t  value) {
	if constexpr (is_same_v<char,uint8_t>) {
		#ifdef BUILD_COMPILER_CLANG //Apparently doesn't reason using `if constexpr` above.
			#pragma clang diagnostic push
			#pragma clang diagnostic ignored "-Wsign-conversion"
		#endif
		string s; s+=value; return s;
		#ifdef BUILD_COMPILER_CLANG
			#pragma clang diagnostic pop
		#endif
	} else return to_string(static_cast<int>(value));
}
inline string to_string( int8_t  value) {
	if constexpr (is_same_v<char, int8_t>) {
		string s; s+=value; return s;
	} else return to_string(static_cast<int>(value));
}
inline string to_string(uint16_t value) { return to_string(static_cast<int>(value)); }
inline string to_string( int16_t value) { return to_string(static_cast<int>(value)); }


} namespace IB { namespace Str {


//Removes any present '\r' characters.  Returns the number removed.
size_t normalize_newlines(char* source);

//Checks `main_string` for `test_string` or `test_char` in the named way.
bool contains  (char const*        main_string, char               test_char  );
bool contains  (std::string const& main_string, char               test_char  );
bool contains  (char const*        main_string, char const*        test_string);
bool contains  (std::string const& main_string, std::string const& test_string);
bool startswith(char const*        main_string, char const*        test_string);
bool startswith(std::string const& main_string, std::string const& test_string);
bool endswith  (char const*        main_string, char const*        test_string);
bool endswith  (std::string const& main_string, std::string const& test_string);

//Searches for any character in `search_chars` in string `main_string` within index range
//	[`min_i`,`max_i`), returning the index of the last occurrence.  Returns `std::string::npos` iff
//	no occurrence was found.
size_t find_last_of     (std::string const& main_string, char const* search_chars, size_t min_i,size_t max_i);
//Searches for any character in `search_chars` in string `main_string` within index range
//	[`min_i`,`max_i`), returning the index of the first character in the last sequence of
//	occurrences.  Returns `std::string::npos` iff no sequence of occurrences was found.
size_t find_firstlast_of(std::string const& main_string, char const* search_chars, size_t min_i,size_t max_i);

//Conversions from strings
template <typename T> T  string_to(std:: string const& str); //{ std:: istringstream i(str); T t; if (!(i>>t)) { return 0; } return t; }
template <typename T> T wstring_to(std::wstring const& str); //{ std::wistringstream i(str); T t; if (!(i>>t)) { return 0; } return t; }
#define LIBIB_SPEC_STRING_TO(TYPE,FUNC)\
	template <> inline TYPE  string_to(std:: string const& str) { return static_cast<TYPE>(std::FUNC(str,nullptr)); }\
	template <> inline TYPE wstring_to(std::wstring const& str) { return static_cast<TYPE>(std::FUNC(str,nullptr)); }
LIBIB_SPEC_STRING_TO(float,      stof  )
LIBIB_SPEC_STRING_TO(double,     stod  )
LIBIB_SPEC_STRING_TO(long double,stold )
LIBIB_SPEC_STRING_TO(uint8_t,    stoul )
LIBIB_SPEC_STRING_TO( int8_t,    stol  )
LIBIB_SPEC_STRING_TO(uint16_t,   stoul )
LIBIB_SPEC_STRING_TO( int16_t,   stol  )
LIBIB_SPEC_STRING_TO(uint32_t,   stoul )
LIBIB_SPEC_STRING_TO( int32_t,   stol  )
LIBIB_SPEC_STRING_TO(uint64_t,   stoull)
LIBIB_SPEC_STRING_TO( int64_t,   stoll )
#undef LIBIB_SPEC_STRING_TO
template <> inline std::wstring  string_to(std:: string const& str) { return std::to_wstring(str); }
template <> inline std:: string wstring_to(std::wstring const& str) { return std::to_string (str); }

//Replacements
std::string get_replaced(std::string const& str_input, std::string const& str_to_find, std::string const& str_to_replace_with);
void        replace     (std::string*       str_input, std::string const& str_to_find, std::string const& str_to_replace_with);

//Gets copy of string with ['A','Z'] changed to ['a','z'] and all other characters the same.
inline std::string to_lower(std::string const& str) {
	std::string tmp = str;
	for (char& c : tmp) if (c>='A'&&c<='Z') c=static_cast<char>(c+'a'-'A');
	return tmp;
}
//Gets copy of string with ['a','z'] changed to ['A','Z'] and all other characters the same.
inline std::string to_upper(std::string const& str) {
	std::string tmp = str;
	for (char& c : tmp) if (c>='a'&&c<='z') c=static_cast<char>(c-'a'+'A');
	return tmp;
}

//Splits a string `main_string` at every occurrence of `test_string`, up to `max_splits` times (`-1`
//	for unlimited).
//	Note: if two `test_string`s are next to each other, there is considered to be an empty string
//		between them.
std::deque<std::string> get_split(std::string const& main_string, std::string const& test_string, int max_splits=-1);

//Returns a version of the string `main_string` that, when printed from cursor x-coordinate
//	`start_cursor_x` on a terminal of width `wrap_width`, will wrap reasonably, with wraps at
//	whitespace or hyphens.  Wraps are additionally re-indented to `wrap_indent` spaces.  The string
//	can contain manual newlines, which will also be respected.
std::string get_wrapped(std::string const& main_string, size_t wrap_width,size_t start_cursor_x=0,size_t wrap_indent=0);

//Remove leading/trailing whitespace.
void trim(          std::string * main_string);
void trim(std::list<std::string>* string_list);
std::string get_trimmed(std::string const& main_string);


}}
#endif
