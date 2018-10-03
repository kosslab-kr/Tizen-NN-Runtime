// Generated file (from: fully_connected_float_1_nnfw.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type4(Type::INT32, {});
  OperandType type3(Type::TENSOR_FLOAT32, {1, 1});
  OperandType type1(Type::TENSOR_FLOAT32, {1, 24});
  OperandType type0(Type::TENSOR_FLOAT32, {1, 3, 4, 2});
  OperandType type2(Type::TENSOR_FLOAT32, {1});
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type1);
  auto b0 = model->addOperand(&type2);
  auto op3 = model->addOperand(&type3);
  auto act_relu = model->addOperand(&type4);
  // Phase 2, operations
  static float op2_init[] = {-0.25449711f, 0.0f, -2.1247749f, 0.0f, -1.143796f, 0.0f, -1.0299346f, 0.0f, -2.2373879f, 0.0f, -0.083096743f, 0.0f, -1.3230739f, 0.0f, 0.15294921f, 0.0f, -0.53045893f, 0.0f, -0.46075189f, 0.0f, -1.4482396f, 0.0f, -1.609534f, 0.0f};
  model->setOperandValue(op2, op2_init, sizeof(float) * 24);
  static float b0_init[] = {0.70098364f};
  model->setOperandValue(b0, b0_init, sizeof(float) * 1);
  static int32_t act_relu_init[] = {0};
  model->setOperandValue(act_relu, act_relu_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_FULLY_CONNECTED, {op1, op2, b0, act_relu}, {op3});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {op1},
    {op3});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
