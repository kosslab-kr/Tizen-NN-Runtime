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

#ifndef __NEURUN_GRAPH_OPERAND_LOWER_INFO_H__
#define __NEURUN_GRAPH_OPERAND_LOWER_INFO_H__

#include <stdint.h>

#include "LayoutSet.h"

namespace neurun
{
namespace graph
{
namespace operand
{

class LowerInfo
{
public:
  class Shape4D
  {
  public:
    Shape4D(uint32_t n, uint32_t h, uint32_t w, uint32_t c) : _n{n}, _h{h}, _w{w}, _c{c}
    {
      // DO NOTHING
    }

  public:
    uint32_t n(void) const { return _n; }
    uint32_t h(void) const { return _h; }
    uint32_t w(void) const { return _w; }
    uint32_t c(void) const { return _c; }

  private:
    uint32_t _n;
    uint32_t _h;
    uint32_t _w;
    uint32_t _c;
  };

public:
  LowerInfo(const Shape4D &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  const Shape4D &shape(void) const { return _shape; }
  const LayoutSet &def_layouts(void) const { return _def_layouts; }
  const LayoutSet &use_layouts(void) const { return _use_layouts; }

public:
  void addDefLayout(const Layout &layout) { _def_layouts.add(layout); }
  void addUseLayout(const Layout &layout) { _use_layouts.add(layout); }

private:
  Shape4D _shape;
  LayoutSet _def_layouts;
  LayoutSet _use_layouts;
};

} // namespace operand
} // namespace graph
} // namespace neurun

#endif // __NEURUN_GRAPH_OPERAND_LOWED_INFO_H__
