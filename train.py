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

import sys
import os
import traceback
import time
import numpy as np
from argparse import ArgumentParser
from singa import tensor, device, optimizer
from singa import utils, metric
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType

import model
import data


def main():
    '''Command line options'''
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Train Alexnet over CIFAR10")

        parser.add_argument('-p', '--port', default=9999, help='listening port')
        parser.add_argument('-C', '--use_cpu', action="store_true")
        parser.add_argument('--max_epoch', default=140)

        # Process arguments
        args = parser.parse_args()
        port = args.port

        use_cpu = args.use_cpu
        if use_cpu:
            print "runing with cpu"
            dev = device.get_default_device()
        else:
            print "runing with gpu"
            dev = device.create_cuda_gpu()

        # start to train
        net = model.create_net(use_cpu)
        agent = Agent(port)
        train(net, dev, agent, args.max_epoch)
        #wait the agent finish handling http request
        agent.stop()
    except SystemExit:
        return
    except:
        #p.terminate()
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")

def initialize(net, dev, opt):
    '''initialize all parameters in the model'''
    print 'Start intialization............'
    for (p, specs) in zip(net.param_values(), net.param_specs()):
        filler = specs.filler
        if filler.type == 'gaussian':
            p.gaussian(filler.mean, filler.std)
        else:
            p.set_value(0)
        opt.register(p, specs)
        print specs.name, filler.type, p.l1()
    net.to_device(dev)
    print 'End intialization............'


def get_data():
    '''load data'''
    data.prepare_train_files()

    train_x, train_y = data.load_train_data()
    test_x, test_y = data.load_test_data()
    mean = data.load_mean_data()
    if mean is None:
        mean = np.average(train_x, axis=0)
        data.save_mean_data(mean)
    train_x -= mean
    test_x -= mean
    return train_x, train_y, test_x, test_y


def handle_cmd(agent):
    pause = False
    stop = False
    while not stop:
        key, val = agent.pull()
        if key is not None:
            msg_type = MsgType.parse(key)
            if msg_type.is_command():
                if MsgType.kCommandPause.equal(msg_type):
                    agent.push(MsgType.kStatus,"Success")
                    pause = True
                elif MsgType.kCommandResume.equal(msg_type):
                    agent.push(MsgType.kStatus, "Success")
                    pause = False
                elif MsgType.kCommandStop.equal(msg_type):
                    agent.push(MsgType.kStatus,"Success")
                    stop = True
                else:
                    agent.push(MsgType.kStatus,"Warning, unkown message type")
                    print "Unsupported command %s" % str(msg)
        if pause and not stop:
            time.sleep(0.1)
        else:
            break
    return stop


def get_lr(epoch):
    '''change learning rate as epoch goes up'''
    if epoch < 120:
        return 0.001
    elif epoch < 130:
        return 0.0001
    else:
        return 0.00001


def train(net, dev, agent, max_epoch, batch_size=100):
    agent.push(MsgType.kStatus, 'Downlaoding data...')
    train_x, train_y, test_x, test_y = get_data()
    print 'training shape', train_x.shape, train_y.shape
    print 'validation shape', test_x.shape, test_y.shape
    agent.push(MsgType.kStatus, 'Finish downloading data')

    opt = optimizer.SGD(momentum=0.9, weight_decay=0.0005)
    accuracy = metric.Accuracy()

    initialize(net, dev, opt)

    tx = tensor.Tensor((batch_size, 3, 32, 32), dev)
    ty = tensor.Tensor((batch_size, ), dev, core_pb2.kInt)
    num_train_batch = train_x.shape[0] / batch_size
    num_test_batch = test_x.shape[0] / (batch_size)

    idx = np.arange(train_x.shape[0], dtype=np.int32)

    for epoch in range(max_epoch):
        if handle_cmd(agent):
            break
        np.random.shuffle(idx)
        print 'Epoch %d' % epoch


        loss, acc = 0.0, 0.0
        for b in range(num_test_batch):
            x = test_x[b * batch_size:(b + 1) * batch_size]
            y = test_y[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            acc += a
        print 'testing loss = %f, accuracy = %f' % (loss / num_test_batch,
                                                    acc / num_test_batch)
        # put test status info into a shared queue
        info = dict(
            phase='test',
            step = epoch,
            accuracy = acc / num_test_batch,
            loss = loss / num_test_batch,
            timestamp = time.time())
        agent.push(MsgType.kInfoMetric, info)

        loss, acc = 0.0, 0.0
        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
            y = train_y[idx[b * batch_size:(b + 1) * batch_size]]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            for (s, p, g) in zip(net.param_specs(),
                                 net.param_values(), grads):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p,
                                       str(s.name))
            info = 'training loss = %f, training accuracy = %f' % (l, a)
            utils.update_progress(b * 1.0 / num_train_batch, info)
        # put training status info into a shared queue
        info = dict(
            phase='train',
            step= epoch,
            accuracy = acc / num_train_batch,
            loss = loss / num_train_batch,
            timestamp = time.time())
        agent.push(MsgType.kInfoMetric, info)
        info = 'training loss = %f, training accuracy = %f' \
            % (loss / num_train_batch, acc / num_train_batch)
        print info

        if epoch > 0 and epoch % 30 == 0:
            net.save(os.path.join(data.parameter_folder, 'parameter_%d' % epoch))
    net.save(os.path.join(data.parameter_folder, 'parameter_last'))


if __name__ == '__main__':
    main()
