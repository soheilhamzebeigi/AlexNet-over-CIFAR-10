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
import urllib, traceback
import numpy as np

from multiprocessing import Process, Queue
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

from singa import tensor, device, optimizer
from singa import utils, initializer, metric, image_tool
from singa.proto import core_pb2

from rafiki.agent import Agent, MsgType

import data
import model

top_k = 5

tool = image_tool.ImageTool()
small_size = 35 
big_size = 45 
crop_size = 32 
num_augmentation = 10

def image_transform(image):
    '''Input an image path and return a set of augmented images (type Image)'''
    global tool
    return tool.load(image).resize_by_list([(small_size + big_size)/2]).crop5(
            (crop_size, crop_size), 5).flip(2).get()

def main(argv=None):
    '''Command line options'''
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

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
            default="parameter",
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
        agent = Agent(port)
        service = Service(agent,use_cpu)
        service.initialize(parameter_file)
        service.serve()
        #wait the agent finish handling http request
        time.sleep(1)
        agent.stop()

    except SystemExit:
        return 
    except:
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2

class Service():
    def __init__(self, agent, use_cpu):
        self.model =model.create(use_cpu)
        if use_cpu:
            print "running with cpu"
            self.device = device.get_default_device()
            #print "cpu mode is not supported at present!"
        else:
            print "runing with gpu"
            self.device = device.create_cuda_gpu()
        self.opt = optimizer.SGD(momentum=0.9, weight_decay=0.0005)
        self.agent = agent

    def initialize(self, parameter_file=None):
        '''get parameters of the model to run the model in predict manner'''
        data.serve_file_prepare()
        print 'Start intialization............'
        parameter = data.get_parameter(parameter_file, True)
        print 'initialize with %s' % parameter
        self.model.load(parameter)
        self.model.to_device(self.device)
        print 'End intialization............'
        self.mean = data.load_mean_data()

    def serve(self):
        '''predict the label for the uploaded images'''
        while True:
            msg,data= self.agent.pull()
            if msg == None:
                continue
            msg=MsgType.parse(msg)
            if msg.is_request():
                try:
                    response = ""
                    images = []
                    for im in image_transform(data):
                        ary = np.array(im.convert('RGB'), dtype=np.float32)
                        images.append(ary.transpose(2, 0, 1))
                    images = np.array(images)
                    #normalize
                    images -= self.mean

                    x = tensor.from_numpy(images.astype(np.float32))
                    x.to_device(self.device)
                    y = self.model.predict(x)
                    y.to_host()
                    y = tensor.to_numpy(y)
                    prob = np.average(y, 0)
                    #sort and reverse
                    labels = np.flipud(np.argsort(prob))
                    for i in range(top_k):
                        response += "%s:%s<br/>" % (get_name(labels[i]),
                                                    prob[labels[i]])
                except:
                    traceback.print_exc()
                    response="sorry, system error."
                self.agent.push(MsgType.kResponse,response)
            elif msg.is_command():
                if MsgType.kCommandStop.equal(msg):
                    print 'get stop command'
                    self.agent.push(MsgType.kStatus,"success")
                    break
                else:
                    print 'get unsupported command %s' % str(msg)
                    self.agent.push(MsgType.kStatus,"failure")
            else:
                print 'get  unsupported message %s' % str(msg)
                self.agent.push(MsgType.kStatus,"failure")
                break
            time.sleep(0.01)
            # while loop
        print "server stop"
        return

label_map = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def get_name(label):
    return label_map[label]


if __name__ == '__main__':
    main()
