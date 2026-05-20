#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, unsigned r>
inline static void vadd_template( T_in const * __restrict ptr_in0,
                                  T_in const * __restrict ptr_in1,
                                  T_out      * __restrict ptr_out ) {
  // define vectors
  aie::vector<T_in,  r> v_in0;
  aie::vector<T_in,  r> v_in1;
  aie::vector<T_out, r> v_out;

  // load data
  v_in0 = aie::load_v<r>(ptr_in0);
  v_in1 = aie::load_v<r>(ptr_in1);

  // element-wise addition using the AIE-API
  v_out = aie::add(v_in0, v_in1);

  // store data
  aie::store_v(ptr_out, v_out);

  return;
}

extern "C" {
  void vadd( bfloat16 const * ptr_in0,
             bfloat16 const * ptr_in1,
             bfloat16       * ptr_out ) {
    vadd_template<bfloat16, bfloat16, 64>(ptr_in0, ptr_in1, ptr_out);
  }
}
