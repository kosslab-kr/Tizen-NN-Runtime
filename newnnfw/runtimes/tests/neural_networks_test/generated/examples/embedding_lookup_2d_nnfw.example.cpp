// Generated file (from: embedding_lookup_2d_nnfw.mod.py). Do not edit
// Begin of an example
{
//Input(s)
{ // See tools/test_generator/include/TestHarness.h:MixedTyped
  // int -> FLOAT32 map
  {{1, {0.0f, 0.1f, 1.0f, 1.1f, 2.0f, 2.1f}}},
  // int -> INT32 map
  {{0, {1, 0, 2}}},
  // int -> QUANT8_ASYMM map
  {}
},
//Output(s)
{ // See tools/test_generator/include/TestHarness.h:MixedTyped
  // int -> FLOAT32 map
  {{0, {1.0f, 1.1f, 0.0f, 0.1f, 2.0f, 2.1f}}},
  // int -> INT32 map
  {},
  // int -> QUANT8_ASYMM map
  {}
}
}, // End of an example
