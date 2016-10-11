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
import urllib, traceback
from singa import tensor, device, optimizer
from singa import utils, initializer, metric
from singa.proto import core_pb2
import data
import process

top_k=5

class Service():
    def __init__(self, model, use_cpu):
        self.model = model
        if use_cpu:
            print "cpu mode is not supported at present!"
        else:
            print "runing with gpu"
            self.device = device.create_cuda_gpu()
        self.opt = optimizer.SGD(momentum=0.9, weight_decay=0.0005)

    def initialize(self,parameter_file=None):
        data.serve_file_prepare()
        print 'Start intialization............'
        parameter = data.get_parameter(parameter_file,True)
        print 'initialize with %s' % parameter
        self.model.load(parameter)
        self.model.to_device(self.device)
        print 'End intialization............'
        self.mean = data.load_mean_data()

    def serve(self,request):
        image = request.files['image']
        if not image:
            return "error, no image file found!"
        if not allowed_file(image.filename):
            return "error, only jpg image is allowed."
        try:
            #process images
            images=process.process_img(image,36,(32,32),True)
            images=np.array(images[0:10]).astype(np.float32)
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
            response =""
            for i in range(top_k):
                response += "%s:%s<br/>" % (get_name(labels[i]),prob[labels[i]])
            return response
        except Exception as e:
            traceback.print_exc()
            print e
            return "sorry, system error."
    
def get_lr(epoch):
    if epoch < 360:
        return 0.0008
    elif epoch < 540:
        return 0.0001
    else:
        return 0.00001
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ["jpg","JPG","JPEG","jpeg"]

label_map={
0:'airplane',
1:'automobile',
2:'bird',
3:'cat',
4:'deer',
5:'dog',
6:'frog',
7:'horse',
8:'ship',
9:'truck'
}
def get_name(label):
    return label_map[label]