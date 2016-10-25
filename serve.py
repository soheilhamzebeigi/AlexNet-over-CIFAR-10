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
import time
import traceback
import numpy as np

from argparse import ArgumentParser

from singa import tensor, device
from singa import image_tool

from rafiki.agent import Agent, MsgType

import data
import model

top_k = 5

tool = image_tool.ImageTool()
num_augmentation = 10


def image_transform(image):
    '''Input an image path and return a set of augmented images (type Image)'''
    global tool
    return tool.load(image).resize_by_list([40]).crop5((32, 32), 5).get()


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


def serve(agent, use_cpu, parameter_file):
    net = model.create_net(use_cpu)
    if use_cpu:
        print "running with cpu"
        dev = device.get_default_device()
    else:
        print "runing with gpu"
        dev = device.create_cuda_gpu()
    agent = agent

    data.prepare_serve_files()
    print 'Start intialization............'
    parameter = data.get_parameter(parameter_file, True)
    print 'initialize with %s' % parameter
    net.load(parameter)
    net.to_device(dev)
    print 'End intialization............'
    mean = data.load_mean_data()

    while True:
        key, val = agent.pull()
        if key is None:
            time.sleep(0.1)
            continue
        msg_type = MsgType.parse(key)
        if msg_type.is_request():
            try:
                response = ""
                images = []
                for im in image_transform(val):
                    ary = np.array(im.convert('RGB'), dtype=np.float32)
                    images.append(ary.transpose(2, 0, 1) - mean)
                images = np.array(images)

                x = tensor.from_numpy(images.astype(np.float32))
                x.to_device(dev)
                y = net.predict(x)
                y.to_host()
                y = tensor.to_numpy(y)
                prob = np.average(y, 0)
                # sort and reverse
                labels = np.flipud(np.argsort(prob))
                for i in range(top_k):
                    response += "%s:%s<br/>" % (get_name(labels[i]),
                                                prob[labels[i]])
            except:
                traceback.print_exc()
                response = "Sorry, system error during prediction."
            agent.push(MsgType.kResponse, response)
        elif MsgType.kCommandStop.equal(msg_type):
                print 'get stop command'
                agent.push(MsgType.kStatus, "success")
                break
        else:
            print 'get unsupported message %s' % str(msg_type)
            agent.push(MsgType.kStatus, "Unknown command")
            break
        # while loop
    print "server stop"


def main():
    try:
        # Setup argument parser
        parser = ArgumentParser(description="SINGA CIFAR SVG TRANING MODEL")

        parser.add_argument("-p", "--port", default=9999, help="listen port")
        parser.add_argument("-C", "--use_cpu", action="store_true")
        parser.add_argument("--parameter_file", help="relative path")

        # Process arguments
        args = parser.parse_args()
        port = args.port

        # start to train
        agent = Agent(port)
        serve(agent, args.use_cpu, args.parameter_file)
        agent.stop()

    except SystemExit:
        return
    except:
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2


if __name__ == '__main__':
    main()
