/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NEURUN_GRAPH_OPERATION_NODE_VISITOR_H__
#define __NEURUN_GRAPH_OPERATION_NODE_VISITOR_H__

#include "Conv2D.h"
#include "MaxPool2D.h"
#include "AvgPool2D.h"
#include "Concat.h"
#include "Reshape.h"
#include "FullyConnected.h"
#include "Softmax.h"
#include "NOP.h"
#include "Permute.h"
#include "Add.h"

namespace neurun
{
namespace graph
{
namespace operation
{

struct NodeVisitor
{
  virtual ~NodeVisitor() = default;

  virtual void visit(const Conv2D::Implicit::Node &) = 0;
  virtual void visit(const MaxPool2D::Implicit::Node &) = 0;
  virtual void visit(const AvgPool2D::Implicit::Node &) = 0;
  virtual void visit(const Concat::Node &) = 0;
  virtual void visit(const Reshape::Node &) = 0;
  virtual void visit(const FullyConnected::Node &) = 0;
  virtual void visit(const Softmax::Node &) = 0;
  virtual void visit(const NOP::Node &) = 0;
  virtual void visit(const Permute::Node &) = 0;
  virtual void visit(const Add::Node &) = 0;
};

} // namespace operation
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERATION_NODE_VISITOR_H__
