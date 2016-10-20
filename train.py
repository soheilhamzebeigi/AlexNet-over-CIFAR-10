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

import sys, os, traceback
import glob, random, shutil, time
import numpy as np
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from singa import tensor, device, optimizer
from singa import utils, initializer, metric
from singa.proto import core_pb2
from rafiki.agent import Agent, MsgType

import model
import data

def main():
    '''Command line options'''
    argv = sys.argv
    try:
        # Setup argument parser
        parser = ArgumentParser(
            description="SINGA CIFAR SVG TRANING MODEL",
            formatter_class=RawDescriptionHelpFormatter)

        parser.add_argument(
            "-p",
            "--port",
            dest="port",
            default=9999,
            help="the port to listen to, default is 9999")
        parser.add_argument(
            "-param",
            "--parameter",
            dest="parameter",
            help="the parameter file path to be loaded")
        parser.add_argument(
            "-C",
            "--cpu",
            dest="use_cpu",
            action="store_true",
            default=False,
            help="Using cpu or not, default is using gpu")

        # Process arguments
        args = parser.parse_args()
        port = args.port
        parameter_file = args.parameter
        use_cpu = args.use_cpu

        # start to train
        m = model.create(use_cpu)
        agent = Agent(port)
        trainer = Trainer(m,agent,use_cpu)
        trainer.initialize(parameter_file)
        trainer.train()
        #wait the agent finish handling http request
        time.sleep(1)
        agent.stop()

    except SystemExit:
        return
    except:
        #p.terminate()
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2

class Trainer():
    '''train a singa model'''

    def __init__(self, model, agent, use_cpu):
        self.model = model
        if use_cpu:
            print "runing with cpu"
            self.device = device.get_default_device()
            #raise CLIError("Currently cpu is not support!")
        else:
            print "runing with gpu"
            self.device = device.create_cuda_gpu()
        self.opt = optimizer.SGD(momentum=0.9, weight_decay=0.0005)
        self.agent = agent 

    def initialize(self, parameter_file):
        '''initialize all parameters in the model'''
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
        '''load data'''
        data.train_file_prepare()

        self.train_x, self.train_y = data.load_train_data()
        self.test_x, self.test_y = data.load_test_data()
        self.mean = data.load_mean_data()
        if self.mean is None:
            self.mean = np.average(self.train_x, axis=0)
            data.save_mean_data(self.mean)
        self.train_x -= self.mean
        self.test_x -= self.mean
        
    def pause(self):
        while True:
            msg,data = self.agent.pull()
            if msg == None:
                continue
            msg=MsgType.parse(msg)
            if MsgType.kCommandResume.equal(msg):
                self.agent.push(MsgType.kStatus,"success")
                break
            elif MsgType.kCommandStop.equal(msg):
                self.agent.push(MsgType.kStatus,"success")
                return False
            else:
                self.agent.push(MsgType.kStatus,"warning, nothing happened")
                print "Receive an unsupported command: %s " % str(msg)
                pass
            time.sleep(0.1)
        return True
    
    def listen(self):
        msg,data = self.agent.pull()
        if not msg == None:
            msg=MsgType.parse(msg)
            if msg.is_command():
                if MsgType.kCommandPause.equal(msg):
                    self.agent.push(MsgType.kStatus,"success")
                    if not self.pause():
                        return False 
                elif MsgType.kCommandStop.equal(msg):
                    self.agent.push(MsgType.kStatus,"success")
                    return False 
                else:
                    self.agent.push(MsgType.kStatus,"warning, nothing happened")
                    print "Unsupported command %s" % str(msg)
                    pass
            else:
                pass
        else:
            pass 
        return True

    def train(self, num_epoch=140, batch_size=50):
        '''train and test model'''
        self.data_prepare()
        print 'training shape', self.train_x.shape, self.train_y.shape
        print 'validation shape', self.test_x.shape, self.test_y.shape

        tx = tensor.Tensor((batch_size, 3, 32, 32), self.device)
        ty = tensor.Tensor((batch_size, ), self.device, core_pb2.kInt)

        num_train_batch = self.train_x.shape[0] / batch_size
        num_test_batch = self.test_x.shape[0] / (batch_size)

        accuracy = metric.Accuracy()
        idx = np.arange(self.train_x.shape[0], dtype=np.int32)
        # frequency of gathering training status info

        skip = 20
        stop=False
        for epoch in range(num_epoch):
            if not self.listen():
                stop=True 
            if stop:
                break
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
            # put test status info into a shared queue
            dic = dict(
                phase='test',
                #step = (epoch + 1) * num_train_batch / skip - 1,
                step=epoch * num_train_batch / skip,
                accuracy=acc / num_test_batch,
                loss=loss / num_test_batch,
                timestamp=time.time())

            self.agent.push(MsgType.kInfoMetric,dic)

            for b in range(num_train_batch):
                if not self.listen():
                    stop=True
                    break
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
                # put training status info into a shared queue
                if b % skip == 0:
                    dic = dict(
                        phase='train',
                        step=(epoch * num_train_batch + b) / skip,
                        accuracy=a,
                        loss=l,
                        timestamp=time.time())
                    self.agent.push(MsgType.kInfoMetric,dic)

                # update progress bar
                utils.update_progress(b * 1.0 / num_train_batch, info)
            info = 'training loss = %f, training accuracy = %f' \
                % (loss / num_train_batch, acc / num_train_batch)
            print info
            if epoch > 0 and epoch % 10 == 0:
                self.model.save(
                    os.path.join(data.parameter_folder, 'parameter_%d' %
                                 epoch))
        self.model.save(os.path.join(data.parameter_folder, 'parameter'))
        return


def get_lr(epoch):
    '''change learning rate as epoch goes up'''
    if epoch < 360:
        return 0.0008
    elif epoch < 540:
        return 0.0001
    else:
        return 0.00001


if __name__ == '__main__':
    main()
