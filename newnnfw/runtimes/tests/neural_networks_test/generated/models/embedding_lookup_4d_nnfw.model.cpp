// Generated file (from: embedding_lookup_4d_nnfw.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type2(Type::TENSOR_INT32, {3, 2, 4, 2});
  OperandType type0(Type::TENSOR_INT32, {3});
  OperandType type1(Type::TENSOR_INT32, {5, 2, 4, 2});
  // Phase 1, operands
  auto index = model->addOperand(&type0);
  auto value = model->addOperand(&type1);
  auto output = model->addOperand(&type2);
  // Phase 2, operations
  model->addOperation(ANEURALNETWORKS_EMBEDDING_LOOKUP, {index, value}, {output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {index, value},
    {output});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
