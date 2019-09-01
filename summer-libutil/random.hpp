#pragma once


#include "stdafx.hpp"


namespace Summer {


//PCG-32 source
//	http://www.pcg-random.org/download.html
//	16 bytes state, period 2¹²⁸.
class RandomSourcePCG32 {
	public:
		typedef uint32_t result_type;

	private:
		union {
			struct { uint64_t _state; uint64_t _inc; };
			result_type _state_packed[4];
		};

	public:
		                            __device_host__          RandomSourcePCG32(                ) :
			_state(0x853C49E6748FEA9Bull),
			_inc  (0xDA3E39CB94B95BDBull)
		{}
		template<class TypeSeedSeq> __device_host__ explicit RandomSourcePCG32(TypeSeedSeq& seq) {
			seq.generate(_state_packed.begin(),_state_packed.end());
		}
		~RandomSourcePCG32() = default;

		__device_host__ constexpr static result_type min() { return std::numeric_limits<result_type>::min(); }
		__device_host__ constexpr static result_type max() { return std::numeric_limits<result_type>::max(); }

		//Seed the source.  State and increment must not both be zero.
		__device_host__ void seed(result_type  seed_value=0x748FEA9Bu) {
			#ifndef BUILD_COMPILER_NVCC
			assert_term(seed_value!=0u,"Seed must not be zero!");
			#endif
			_state_packed[0] = _state_packed[1] = _state_packed[2] = _state_packed[3] = seed_value;
		}
		__device_host__ void seed(uint64_t state, uint64_t increment ) {
			#ifndef BUILD_COMPILER_NVCC
			assert_term(state!=0ull||increment!=0ull,"State and increment must not both be zero!");
			#endif
			_state = state;
			_inc   = increment;
		}

		__device_host__ result_type operator()() {
			result_type xorshifted = static_cast<result_type>( ((_state>>18u)^_state) >> 27u );
			int rot = static_cast<int>( _state >> 59u ); //top five bits
			result_type result = ( xorshifted >> rot ) | ( xorshifted << ((-rot)&31) );
			_state = _state*6364136223846793005ull + _inc;
			return result;
		}
		__device_host__ void discard(unsigned long long count) {
			for (unsigned long long i=0;i<count;++i) {
				_state = _state*6364136223846793005ull + _inc;
			}
		}
};


template<class TypeSource> class RandomNumberGenerator {
	private:
		TypeSource _rng;

	public:
		//Seeds RNG.  If no argument is specified, use current time.  Note: similar seeds sometimes
		//	result in similar random numbers generated early-on; consider hashing your prospective
		//	seeds first.
		void seed(             ) {
			seed(static_cast<uint32_t>( time(nullptr) ));
		}
		void seed(uint32_t seed) {
			//This cast should hopefully be superfluous.
			_rng.seed(static_cast<typename TypeSource::result_type>(seed));
		}

		__device_host__ float get_uniform() {
			uint32_t val = _rng();
			return static_cast<float>(val) / static_cast<float>(std::numeric_limits<uint32_t>::max());
		}
		__device_host__ Vec2f get_disk() {
			float angle = TAU * get_uniform();
			float radius = std::sqrt(get_uniform());
			return radius*Vec2f( std::cos(angle), std::sin(angle) );
		}
		__device_host__ Vec3f get_coshemi(Vec3f const& normal) {
			float disk_angle = TAU * get_uniform();

			float disk_radius_sq = get_uniform();
			float disk_radius = std::sqrt(disk_radius_sq);

			float radicand = 1.0f - disk_radius_sq;
			float height = std::sqrt(radicand);

			Vec3f frame_x, frame_y;
			build_frame(normal,&frame_x,&frame_y);
			return
				disk_radius*std::cos(disk_angle) * frame_x +
				disk_radius*std::sin(disk_angle) * frame_y +
				height                           * normal
			;
		}
};

typedef RandomNumberGenerator<RandomSourcePCG32> RNG;


}
