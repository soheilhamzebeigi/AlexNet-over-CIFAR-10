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
import os, sys, shutil
import urllib
import cPickle
import numpy as np

data_folder = "data_"
tar_data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
tar_data_name = 'cifar-10-python.tar.gz'
data_path = 'cifar-10-batches-py'

parameter_folder = "parameter_"
parameter_name = "parameter"
tar_parameter_url = "http://comp.nus.edu.sg/~dbsystem/singa/assets/file/parameter.tar.gz"
tar_parameter_name = 'parameter.tar.gz'

mean_url = "http://comp.nus.edu.sg/~dbsystem/singa/assets/file/train.mean.npy"
mean_name = "train.mean.npy"


def load_dataset(filepath):
    '''load data from binary file'''
    print 'Loading data file %s' % filepath
    with open(filepath, 'rb') as fd:
        cifar10 = cPickle.load(fd)
    image = cifar10['data'].astype(dtype=np.uint8)
    image = image.reshape((-1, 3, 32, 32))
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    return image, label


def load_train_data(num_batches=5):
    labels = []
    batchsize = 10000
    images = np.empty((num_batches * batchsize, 3, 32, 32), dtype=np.uint8)
    for did in range(1, num_batches + 1):
        fname_train_data = os.path.join(data_folder, data_path,
                                        "data_batch_{}".format(did))
        image, label = load_dataset(fname_train_data)
        images[(did - 1) * batchsize:did * batchsize] = image
        labels.extend(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def load_test_data():
    images, labels = load_dataset(
        os.path.join(data_folder, data_path, "test_batch"))
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def load_mean_data():
    mean_path = os.path.join(data_folder, mean_name)
    if os.path.exists(mean_path):
        return np.load(mean_path)
    return None


def save_mean_data(mean):
    mean_path = os.path.join(data_folder, mean_name)
    np.save(mean_path, mean)
    return


def train_file_prepare():
    '''download train file'''
    if os.path.exists(os.path.join(data_folder, data_path)):
        return

    print "download file"
    #clean data
    download_file(tar_data_url, data_folder)
    untar_data(os.path.join(data_folder, tar_data_name), data_folder)


def serve_file_prepare():
    '''download parameter file and mean file'''
    if not os.path.exists(os.path.join(parameter_folder, parameter_name)):
        print "download parameter file"
        download_file(tar_parameter_url, parameter_folder)
        untar_data(
            os.path.join(parameter_folder, tar_parameter_name),
            parameter_folder)

    if not os.path.exists(os.path.join(data_folder, mean_name)):
        print "download mean file"
        download_file(mean_url, data_folder)
    #clean data


def download_file(url, dest):
    '''
    download one file to dest
    '''
    if not os.path.exists(dest):
        os.makedirs(dest)
    if (url.startswith('http')):
        file_name = url.split('/')[-1]
        target = os.path.join(dest, file_name)
        urllib.urlretrieve(url, target)
    return


def get_parameter(file_name=None, auto_find=False):
    '''
    get a parameter file or return none
    '''
    if not os.path.exists(parameter_folder):
        os.makedirs(parameter_folder)
        return None

    if file_name:
        return os.path.join(parameter_folder, file_name)

    #find the last parameter file if outo_find is True
    if auto_find:
        parameter_list = []
        for f in os.listdir(parameter_folder):
            if f.endswith(".model"):
                parameter_list.append(os.path.join(parameter_folder, f[0:-6]))
        if len(parameter_list) == 0:
            return None
        parameter_list.sort()
        return parameter_list[-1]
    else:
        return None


def untar_data(file_path, dest):
    print 'untar data ..................'
    tar_file = file_path
    import tarfile
    tar = tarfile.open(tar_file)
    print dest
    print file_path
    tar.extractall(dest)
    tar.close()
