// Generated file (from: topk_v2_2D_quant8.mod.py). Do not edit
// Begin of an example
{
//Input(s)
{ // See tools/test_generator/include/TestHarness.h:MixedTyped
  // int -> FLOAT32 map
  {},
  // int -> INT32 map
  {},
  // int -> QUANT8_ASYMM map
  {{0, {3, 4, 5, 6, 7, 8, 9, 1, 2, 18, 19, 11}}}
},
//Output(s)
{ // See tools/test_generator/include/TestHarness.h:MixedTyped
  // int -> FLOAT32 map
  {},
  // int -> INT32 map
  {{1, {3, 2, 2, 1, 2, 1}}},
  // int -> QUANT8_ASYMM map
  {{0, {6, 5, 9, 8, 19, 18}}}
}
}, // End of an example
