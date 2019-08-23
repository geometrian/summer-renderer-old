#include "string.hpp"

#include <cctype> //`std::isspace(...)`.
#include <cstring> //`strlen(...)`.

#include <algorithm> //`std::min(...)`.


namespace std {


string  to_string (wstring const& str) {
	string result;
	result.resize(str.length()*MB_CUR_MAX); //Worst case length (all maximum-length characters)

	size_t numbytes = wcstombs( result.data(), str.c_str(),str.length() ); //Convert
	assert_term(numbytes!=~size_t(0),"Failed to convert character!");

	result.resize(numbytes); //Shrink to fit

	return result;
}
wstring to_wstring( string const& str) {
	wstring result;
	result.resize(str.length()); //Worst case length (no Unicode characters)

	size_t numwchars = mbstowcs( result.data(), str.c_str(),str.length() ); //Convert
	assert_term(numwchars!=~size_t(0),"Failed to convert character!");

	result.resize(numwchars); //Shrink to fit

	return result;
}


} namespace IB { namespace Str {


size_t normalize_newlines(char* source) {
	//First loop skips until the first '\r' is reached.  If found, jumps into second loop,
	//	which shifts over characters in "source", in-place.

	char c;
	size_t index_read=0, index_write;

	LOOP1: {
		char c = source[index_read];
		switch (c) {
			default:
				++index_read;
				goto LOOP1;
			case '\r':
				index_write = index_read;
				goto CASE_R;
			case '\0':
				return 0; //This string contains no '\r' characters to remove!
		}
	}

	LOOP2: {
		c = source[index_read];
		switch (c) {
			default:
				source[index_write++] = c;
				//Fallthrough
			CASE_R: case '\r':
				++index_read;
				goto LOOP2;
			case '\0':
				break;
		}
	}

	source[index_write] = '\0';

	assert_term(index_read>=index_write,"Implementation error!");
	return index_read - index_write;
}

bool contains  (char const*        main_string, char               test_char  ) {
	LOOP:
		char c = *main_string;
		if (c==test_char) return true;
		else if (c!='\0') {
			++main_string;
			goto LOOP;
		}
	return false;
}
bool contains  (std::string const& main_string, char               test_char  ) {
	for (char c : main_string) {
		if (c == test_char) return true;
	}
	return false;
}
bool contains  (char const*        main_string, char const*        test_string) {
	size_t len_main = strlen(main_string);
	size_t len_test = strlen(test_string);
	for (size_t shift=0;shift<len_main-len_test+1;++shift) {
		if (memcmp(main_string,test_string,len_test)!=0);
		else return true;
	}
	return false;
}
bool contains  (std::string const& main_string, std::string const& test_string) {
	return main_string.find(test_string) != std::string::npos;
}
bool startswith(char const*        main_string, char const*        test_string) {
	//TODO: test which is faster
	#if 0
		while (*test_string!='\0') {
			if (*main_string!='\0') {
				if (*main_string==*test_string) {
					++main_string;
					++test_string;
				} else return false;
			} else return false;
		}
		return true;
	#else
		int i = 0;
		LOOP:
			char c1 = main_string[i];
			char c2 = test_string[i];

			if (c2!='\0'); else return true;
			if (c1!='\0' && c1==c2) {
				++i;
				goto LOOP;
			}
		return false;
	#endif
}
bool startswith(std::string const& main_string, std::string const& test_string) {
	if (test_string.length()>main_string.length()) return false;
	//if (main_string.find(test_string)==0) return true;
	//return false;
	return main_string.compare( 0,test_string.size(), test_string );
}
bool endswith  (char const*        main_string, char const*        test_string) {
	size_t len_main = strlen(main_string);
	size_t len_test = strlen(test_string);
	if (len_test<=len_main); else return false;
	return memcmp(main_string+len_main-len_test,test_string,len_test) == 0;
}
bool endswith  (std::string const& main_string, std::string const& test_string) {
	if (test_string.length()<=main_string.length()) {
		//return main_string.rfind(test_string) == main_string.length()-test_string.length();
		return main_string.compare( main_string.length()-test_string.length(),test_string.length(), test_string ) == 0;
	} else {
		return false;
	}
}

size_t find_last_of     (std::string const& main_string, char const* search_chars, size_t min_i,size_t max_i) {
	if (!main_string.empty()) {
		assert_term(min_i<main_string.length()||min_i==std::string::npos,"Minimum index (%zu) must be inside string (length %zu) or else `std::string::npos`!",min_i,main_string.length());
		if (min_i!=std::string::npos);
		else min_i=0;

		assert_term(max_i<=main_string.length()||max_i==std::string::npos,"Maximum index (%zu) must be no larger than the string length (%zu) or else `std::string::npos`!",max_i,main_string.length());
		if (max_i!=std::string::npos);
		else max_i=main_string.length();

		assert_term(min_i<max_i,"Minimum index (%zu) must be smaller than maximum index (%zu)!",min_i,max_i);

		assert_term(
			min_i<main_string.length() && max_i<=main_string.length(),
			"Implementation error!"
		);

		//TODO: edit algorithm so that we don't need to cast to `int`; which is done to prevent
		//	unsigned wraparound.
		assert_term(
			static_cast<size_t>(static_cast<int>(min_i))==min_i &&
			static_cast<size_t>(static_cast<int>(max_i))==max_i,
			"Overflow!"
		);
		for (int i=static_cast<int>(max_i)-1; i>=static_cast<int>(min_i); --i) {
			char c = main_string[static_cast<size_t>(i)];

			if (contains(search_chars,c)) return static_cast<size_t>(i);
		}
	}
	return std::string::npos;
}
size_t find_firstlast_of(std::string const& main_string, char const* search_chars, size_t min_i,size_t max_i) {
	if (!main_string.empty()) {
		assert_term(min_i<main_string.length()||min_i==std::string::npos,"Minimum index (%zu) must be inside string (length %zu) or else `std::string::npos`!",min_i,main_string.length());
		if (min_i!=std::string::npos);
		else min_i=0;

		assert_term(max_i<=main_string.length()||max_i==std::string::npos,"Maximum index (%zu) must be no larger than the string length (%zu) or else `std::string::npos`!",max_i,main_string.length());
		if (max_i!=std::string::npos);
		else max_i=main_string.length();

		assert_term(min_i<max_i,"Minimum index (%zu) must be smaller than maximum index (%zu)!",min_i,max_i);

		assert_term(
			min_i<main_string.length() && max_i<=main_string.length(),
			"Implementation error!"
		);

		//TODO: edit algorithm so that we don't need to cast to `int`; which is done to prevent
		//	unsigned wraparound.
		assert_term(
			static_cast<size_t>(static_cast<int>(min_i))==min_i &&
			static_cast<size_t>(static_cast<int>(max_i))==max_i,
			"Overflow!"
		);
		for (int i=static_cast<int>(max_i)-1; i>=static_cast<int>(min_i); --i) {
			char c = main_string[static_cast<size_t>(i)];

			if (contains(search_chars,c)) {
				//`i` is the last character in the last sequence of occurrences.  Now find the
				//	first of the last.
				LOOP:
					if (i>static_cast<int>(min_i)) {
						if (contains(search_chars,main_string[static_cast<size_t>(i-1)])) {
							--i;
							goto LOOP;
						}
					}
				return static_cast<size_t>(i);
			}
		}
	}
	return std::string::npos;
}

std::string get_replaced(std::string const& str_input, std::string const& str_to_find, std::string const& str_to_replace_with) {
	std::string str_input_copy = str_input;
	replace(&str_input_copy,str_to_find,str_to_replace_with);
	return str_input_copy;
}
void        replace     (std::string*       str_input, std::string const& str_to_find, std::string const& str_to_replace_with) {
	size_t pos = 0;
	size_t find_length = str_to_find.length();
	size_t replace_length = str_to_replace_with.length();
	if (find_length==0) return;
	for (;(pos=str_input->find(str_to_find,pos))!=std::string::npos;) {
		str_input->replace(pos, find_length, str_to_replace_with);
		pos += replace_length;
	}
}

std::deque<std::string> get_split(std::string const& main_string, std::string const& test_string, int max_splits/*=-1*/) {
	//Note: if two "test_string"s are next to each other, puts as "".  TODO: test that.
	//See also http://www.computing.net/answers/programming/string-split-for-c/13456.html
	assert_term(max_splits==-1||max_splits>0,"The maximum number of splits must be positive (or -1 to disable)!");
	assert_term(!test_string.empty(),"Test string must be non-empty!");

	std::deque<std::string> result;

	int num_splits = 0;
	size_t offset = 0;
	LOOP:
		size_t loc = main_string.find(test_string,offset);
		if (loc!=main_string.npos) {
			assert_term(offset<=loc,"Implementation error!");
			result.emplace_back(main_string.substr(offset,loc-offset));
			offset = loc + test_string.size();

			++num_splits;
			if (max_splits==-1 || num_splits<max_splits) goto LOOP;
		}
	result.emplace_back(main_string.substr(offset));

	return result;



	/*bool has_looped = false;
	std::string temp_main_string = main_string;
	size_t cut_at;
	while ( (cut_at=temp_main_string.find_first_of(test_string)) != temp_main_string.npos ) {
		if (cut_at>0) {
			result.push_back(temp_main_string.substr(0,cut_at));
		}
		else if (cut_at==0) result.push_back("");
		temp_main_string = temp_main_string.substr(cut_at+1);
		has_looped = true;
	}
	if (has_looped) {
		result.push_back(temp_main_string);
	}
	if (max_splits!=-1) { //TODO: improve
		size_t s;
		while ((s=result.size())>static_cast<size_t>(max_splits+1)) {
			result[s-2] += test_string + result[s-1];
			result.pop_back();
		}
	}
	if (result.size()==0) result.push_back(main_string);
	return result;*/
}

inline static std::string _get_wrapped_helper(std::string const& line, size_t wrap_width,size_t start_cursor_x,size_t wrap_indent,std::string const& wrap_str) {
	std::string tmp;
	tmp.reserve(2*line.length());

	size_t offset = 0;
	size_t x = start_cursor_x;
	while (offset<line.length()) { CONTINUE_WITHOUT_CHECK:
		size_t left_on_scrn = wrap_width - x;
		assert_term(left_on_scrn>0,"Implementation error!");
		size_t left_to_fmt = line.length() - offset;
		if (left_to_fmt > left_on_scrn) {
			//Need to wrap.
			//	Search for a wrap point.
			size_t wrap_i;
			//		First preference for whitespace or hyphens
			                                     wrap_i=find_firstlast_of(line, " \t-",        offset+1,std::min(offset+left_on_scrn+1,line.length()));
			//		But if we can't find that, try slashes, etc.
			if (wrap_i!=std::string::npos); else wrap_i=find_firstlast_of(line, "/\\()[]{}",   offset+1,std::min(offset+left_on_scrn+1,line.length()));
			//		But if we can't find that either, at least try a symbol.
			if (wrap_i!=std::string::npos); else wrap_i=find_firstlast_of(line, "~!@#$%^&*+=", offset+1,std::min(offset+left_on_scrn+1,line.length()));

			if (wrap_i!=std::string::npos) {
				//Found a good place to wrap, so wrap there.
				assert_term(offset<wrap_i,"Implementation error!");
				if (!std::isspace(line[wrap_i])) {
					size_t to_add = wrap_i - offset + 1;

					tmp += line.substr(offset,to_add);
					x += to_add;
					offset += to_add;
				} else {
					//Don't add the whitespace; no `+1` here.
					size_t to_add = wrap_i - offset;
					tmp += line.substr(offset,to_add);
					x += to_add;
					//However, do need to add it to offset into string
					offset += to_add + 1;
				}

				if (x<wrap_width) {
					tmp += wrap_str;
				} else {
					tmp += wrap_str.substr(1); //Skip the '\n'; the console itself effectively adds it.
				}
				x = wrap_indent;

				//	Eat remaining whitespace in wrap.
				LOOP_EAT_WS:
					if (!std::isspace(line[offset]));
					else { ++offset; goto LOOP_EAT_WS; }
			} else {
				//	Couldn't find anyplace good to wrap at.
				wrap_i = line.find_first_of(" \t-", offset);
				if (wrap_i!=std::string::npos && wrap_indent+wrap_i-offset<wrap_width) {
					//		Wrapping immediately would make the next line okay.  So do that.
					tmp += wrap_str;
					x = wrap_indent;
					goto CONTINUE_WITHOUT_CHECK;
				} else {
					//		Wrapping immediately would not help.  Give up and wrap through the word.
					tmp += line.substr(offset,left_on_scrn);
					tmp += wrap_str.substr(1); //Skip the '\n'; the console itself effectively adds it.
					offset += left_on_scrn;
					x = wrap_indent;
				}
			}
		} else {
			//Don't need to wrap!  The remainder of the line fits.
			tmp += line.substr(offset);
			//offset += line.length()-offset;
			//x      += line.length()-offset;
			break; //Optimization.
		}
	}

	return tmp;
}
std::string get_wrapped(std::string const& main_string, size_t wrap_width,size_t start_cursor_x/*=0*/,size_t wrap_indent/*=0*/) {
	std::string wrap_str = "\n";
	for (size_t i=0;i<wrap_indent;++i) wrap_str+=" ";

	std::string tmp;
	auto lines = get_split(main_string,"\n");
	for (size_t i=0;i<lines.size();++i) {
		tmp += _get_wrapped_helper(lines[i], wrap_width,start_cursor_x,wrap_indent,wrap_str);
		if (i<lines.size()-1) tmp+='\n';
	}

	return tmp;
}

void trim(          std::string * main_string) {
	//TODO: optimize
	while (main_string->length()>0) {
		char first_char = (*main_string)[0];
		if (isspace(first_char));
		else {
			//while (main_string->length()>0) {
			LOOP:
				size_t l = main_string->length();
				char last_char = (*main_string)[l-1];
				if (isspace(last_char));
				else return;
				main_string->erase(l-1,l);
				goto LOOP;
			//}
		}
		main_string->erase(0,1);
	}
}
void trim(std::list<std::string>* string_list) {
	for (auto iter=string_list->begin(); iter!=string_list->end(); ++iter) {
		trim(&(*iter));
	}
}
std::string get_trimmed(std::string const& main_string) {
	std::string result = main_string;
	trim(&result);
	return result;
}

/*bool string_to_number(std::string const& str,    int*__restrict number) {
	if (str.length()==0) return false; //covers the "" case, which would otherwise return 0
	*number = 0;
	bool negate = false;
	for (unsigned int i=0; i<str.length(); ++i) {
		char c = str[i];
		if (c<='9'&&c>='0') { *number *= 10; *number += c-'0'; }
		else if (c=='-'&&i==0) negate = true;
		else if (c=='.') return false;
		else return false;
	}
	if (negate) *number = -(*number);
	return true;
}
bool string_to_number(std::string const& str,  float*__restrict number) {
	if (str.length()==0) return false; //covers the "" case, which would otherwise return 0.0
	*number = 0.0f;
	bool negate = false;
	int decimal_loc = -1;
	for (unsigned int i=0; i<str.length(); ++i) {
		char c = str[i];
		if (c<='9'&&c>='0') { *number *= 10.0f; *number += c-'0'; }
		else if (c=='-'&&i==0) negate = true;
		else if (c=='.') {
			if (decimal_loc==-1) decimal_loc = (int)(str.length()) - i - 1;
			else return false;
		}
		else return false;
	}
	if (decimal_loc!=-1) { for (int i=0;i<decimal_loc;++i) *number /= 10.0f; }
	if (negate) *number = -(*number);
	return true;
}
bool string_to_number(std::string const& str, double*__restrict number) {
	if (str.length()==0) return false; //covers the "" case, which would otherwise return 0.0
	*number = 0.0;
	bool negate = false;
	int decimal_loc = -1;
	for (unsigned int i=0; i<str.length(); ++i) {
		char c = str[i];
		if (c<='9'&&c>='0') { *number *= 10.0; *number += c-'0'; }
		else if (c=='-'&&i==0) negate = true;
		else if (c=='.') {
			if (decimal_loc==-1) decimal_loc = (int)(str.length()) - i - 1;
			else return false;
		}
		else return false;
	}
	if (decimal_loc!=-1) { for (int i=0;i<decimal_loc;++i) *number /= 10.0; }
	if (negate) *number = -(*number);
	return true;
}*/


}}
