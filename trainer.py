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

import sys, glob, os, random, shutil, time
import numpy as np
import urllib 
from singa import tensor, device, optimizer
from singa import utils, initializer, metric
from singa.proto import core_pb2

import data

class Trainer():
    def __init__(self, model, use_cpu, queue):
        self.model = model
        if use_cpu:
            raise CLIError("Currently cpu is not support!")
        else:
            print "runing with gpu"
            self.device = device.create_cuda_gpu()
        self.opt = optimizer.SGD(momentum=0.9, weight_decay=0.0005)
        self.queue = queue

    def initialize(self,parameter_file):
        print 'Start intialization............'
        if parameter_file:
            parameter = data.get_parameter(parameter_file)
            print 'initialize with %s' % parameter
            self.model.load(parameter)
        else:
            for (p, specs) in zip(self.model.param_values(),
                                  self.model.param_specs()):
                filler = specs.filler
                if filler.type == 'gaussian':
                    initializer.gaussian(p, filler.mean, filler.std)
                elif filler.type == 'xavier':
                    initializer.xavier(p)
                    p *= 0.5  # 0.5 if use glorot, which would have val acc to 83
                else:
                    p.set_value(0)
                self.opt.register(p, specs)
                print specs.name, filler.type, p.l1()
        self.model.to_device(self.device)
        print 'End intialization............'

    def data_prepare(self):

        data.train_file_prepare()
        
        self.train_x, self.train_y = data.load_train_data()
        self.test_x, self.test_y = data.load_test_data()
        self.mean = data.load_mean_data()
        if self.mean == None:
            self.mean=np.average(self.train_x, axis=0)
            data.save_mean_data(self.mean)
        self.train_x -= self.mean
        self.test_x -= self.mean

    def train(self, num_epoch=140, batch_size=50):

        self.data_prepare()

        print 'training shape', self.train_x.shape, self.train_y.shape
        print 'validation shape', self.test_x.shape, self.test_y.shape

        tx = tensor.Tensor((batch_size, 3, 32, 32), self.device)
        ty = tensor.Tensor((batch_size, ), self.device, core_pb2.kInt)

        num_train_batch = self.train_x.shape[0] / batch_size
        num_test_batch = self.test_x.shape[0] / (batch_size)

        accuracy = metric.Accuracy()
        idx = np.arange(self.train_x.shape[0], dtype=np.int32)
        skip = 20
        for epoch in range(num_epoch):
            np.random.shuffle(idx)
            loss, acc = 0.0, 0.0
            print 'Epoch %d' % epoch

            loss, acc = 0.0, 0.0
            for b in range(num_test_batch):
                x = self.test_x[b * batch_size:(b + 1) * batch_size]
                y = self.test_y[b * batch_size:(b + 1) * batch_size]
                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)
                l, a = self.model.evaluate(tx, ty)
                loss += l
                acc += a

            print 'testing loss = %f, accuracy = %f' % (loss / num_test_batch,
                                                        acc / num_test_batch)
            dic = dict(
                phase = 'test',
                #step = (epoch + 1) * num_train_batch / skip - 1,
                step = epoch * num_train_batch / skip,
                accuracy = acc / num_test_batch,
                loss = loss / num_test_batch,
                timestamp = time.time()
            )
            self.queue.put(dic)

            for b in range(num_train_batch):
                x = self.train_x[idx[b * batch_size:(b + 1) * batch_size]]
                y = self.train_y[idx[b * batch_size:(b + 1) * batch_size]]
                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)
                grads, (l, a) = self.model.train(tx, ty)
                loss += l
                acc += a
                for (s, p, g) in zip(self.model.param_specs(),
                                     self.model.param_values(), grads):
                    self.opt.apply_with_lr(epoch, get_lr(epoch), g, p,
                                           str(s.name))
                info = 'training loss = %f, training accuracy = %f' % (l, a)
                if b % skip == 0:
                    dic = dict(
                        phase = 'train',
                        step = (epoch * num_train_batch + b) / skip,
                        accuracy = a,
                        loss = l,
                        timestamp = time.time()
                    )
                    self.queue.put(dic)

                # update progress bar
                utils.update_progress(b * 1.0 / num_train_batch, info)
            print ""
            info = 'training loss = %f, training accuracy = %f' \
                % (loss / num_train_batch, acc / num_train_batch)
            print info

            if epoch > 0 and epoch % 10 == 0:
                self.model.save(os.path.join(data.parameter_folder,'parameter_%d' % epoch))
        self.model.save(os.path.join(data.parameter_folder,'parameter'))
        return
    
def get_lr(epoch):
    if epoch < 360:
        return 0.0008
    elif epoch < 540:
        return 0.0001
    else:
        return 0.00001