# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================
from singa import layer
from singa import metric
from singa import loss
from singa import net as ffnet

def add_layer_group(net, name, nb_filers, sample_shape=None):
    net.add(layer.Conv2D(name + '_1',nb_filers,3,1,pad=1,input_sample_shape=sample_shape))
    net.add(layer.Activation(name + 'act_1'))
    net.add(layer.Conv2D(name + '_2', nb_filers, 3, 1, pad=1))
    net.add(layer.Activation(name + 'act_2'))
    net.add(layer.MaxPooling2D(name, 2, 2, pad=0))

def create():
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    add_layer_group(net, 'conv1', 64, (3, 32, 32))
    add_layer_group(net, 'conv2', 128)
    add_layer_group(net, 'conv3', 256)
    add_layer_group(net, 'conv4', 512)
    add_layer_group(net, 'conv5', 512)
    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('ip1', 512))
    net.add(layer.Activation('relu_ip1'))
    net.add(layer.Dropout('drop1'))
    net.add(layer.Dense('ip2', 10))
    return net

