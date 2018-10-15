#include <gtest/gtest.h>

#include "graph/operand/LayoutSet.h"

using neurun::graph::operand::Layout;
using neurun::graph::operand::LayoutSet;

TEST(graph_operand_LayoutSet, layout_set_operators)
{
  LayoutSet set1{Layout::NCHW};
  LayoutSet set2{Layout::NHWC};
  LayoutSet set3 = set1 | set2;

  ASSERT_EQ(set3.size(), 2);

  ASSERT_EQ((set3 - set1).size(), 1);
  ASSERT_EQ((set3 - set1).contains(Layout::NHWC), true);
  ASSERT_EQ((set3 - set2).size(), 1);
  ASSERT_EQ((set3 - set2).contains(Layout::NCHW), true);
  ASSERT_EQ((set3 - set3).size(), 0);

  ASSERT_EQ((set3 & set1).size(), 1);
  ASSERT_EQ((set3 & set1).contains(Layout::NCHW), true);
  ASSERT_EQ((set3 & set2).size(), 1);
  ASSERT_EQ((set3 & set2).contains(Layout::NHWC), true);
  ASSERT_EQ((set1 & set2).size(), 0);
}
