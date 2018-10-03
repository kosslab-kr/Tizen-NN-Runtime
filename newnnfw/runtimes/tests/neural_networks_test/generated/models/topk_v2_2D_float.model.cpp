// Generated file (from: topk_v2_2D_float.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type1(Type::INT32, {});
  OperandType type2(Type::TENSOR_FLOAT32, {3,2});
  OperandType type0(Type::TENSOR_FLOAT32, {3,4});
  OperandType type3(Type::TENSOR_INT32, {3,2});
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto k = model->addOperand(&type1);
  auto op2 = model->addOperand(&type2);
  auto op3 = model->addOperand(&type3);
  // Phase 2, operations
  static int32_t k_init[] = {2};
  model->setOperandValue(k, k_init, sizeof(int32_t) * 1);
  model->addOperationEx(ANEURALNETWORKS_TOPK_V2_EX, {op1, k}, {op2, op3});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {op1},
    {op2, op3});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
