// Generated file (from: cast_ex_int32_to_float32.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type1(Type::TENSOR_FLOAT32, {2, 3});
  OperandType type0(Type::TENSOR_INT32, {2, 3});
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type1);
  // Phase 2, operations
  model->addOperationEx(ANEURALNETWORKS_CAST_EX, {op1}, {op2});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {op1},
    {op2});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
