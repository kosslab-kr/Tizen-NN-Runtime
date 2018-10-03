#!/usr/bin/python

# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import os
import struct

dir_path = os.path.dirname(os.path.realpath(__file__))

builtin_ops = [
    "CONV_2D",
]

# N,H,W,C
input_shape = [1, 6, 6, 3]
kernel_shape = [1, 3, 3, 3]

# load template file.
with open(dir_path + "/CONV_2D.template.json", 'r') as f:
    graph = json.loads(f.read())
    f.close()

tensors = graph['subgraphs'][0]['tensors']
buffers = graph['buffers']

buffer_map = {}

# shape setup
for t in tensors:
    if t['name'] == 'input':
        t['shape'] = input_shape
    elif t['name'] == 'weights':
        t['shape'] = kernel_shape
    elif t['name'] == 'convolution_bias':
        # bias size = N of weight
        t['shape'] = [kernel_shape[0]]
    elif t['name'] == 'output':
        # just for now, the same padding algorithm.
        # stride = 1
        t['shape'][0] = 1  # N
        t['shape'][1] = input_shape[1]  # H
        t['shape'][2] = input_shape[2]  # W
        t['shape'][3] = kernel_shape[0]  # C

    buffer_map[t['buffer']] = {'name': t['name'], 'shape': t['shape']}

# buffer setup
for i in range(len(buffers)):
    if buffer_map[i]['name'] == 'weights':
        shape = buffer_map[i]['shape']

        weight = np.ones(shape)
        n = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]
        for nn in range(n):
            for hh in range(h):
                for ww in range(w):
                    for cc in range(c):
                        if cc == 0:
                            weight[nn][hh][ww][cc] = 1.0
                        else:
                            weight[nn][hh][ww][cc] = 0.0

        weight_list = weight.flatten()
        weight_bytes = struct.pack('%sf' % (len(weight_list)), *weight_list)
        weight_uints = struct.unpack('%sB' % (len(weight_list) * 4), weight_bytes)

        buffers[i]['data'] = list(weight_uints)

    elif buffer_map[i]['name'] == 'convolution_bias':
        # weight of N
        shape = buffer_map[i]['shape']

        bias = np.zeros(shape)
        bias_list = bias.flatten()
        bias_bytes = struct.pack('%sf' % (len(bias_list)), *bias_list)
        bias_uints = struct.unpack('%sB' % (len(bias_list) * 4), bias_bytes)

        buffers[i]['data'] = list(bias_uints)

with open('model.json', 'w') as f:
    f.write(json.dumps(graph, indent=2))
