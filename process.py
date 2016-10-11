#!/usr/bin/env python
#/************************************************************
#*
#* Licensed to the Apache Software Foundation (ASF) under one
#* or more contributor license agreements.  See the NOTICE file
#* distributed with this work for additional information
#* regarding copyright ownership.  The ASF licenses this file
#* to you under the Apache License, Version 2.0 (the
#* "License"); you may not use this file except in compliance
#* with the License.  You may obtain a copy of the License at
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing,
#* software distributed under the License is distributed on an
#* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#* KIND, either express or implied.  See the License for the
#* specific language governing permissions and limitations
#* under the License.
#*
#*************************************************************/

#**************
#*sudo apt-get install libjpeg-dev
#*sudo pip install 

from PIL import Image
import sys, glob, os, random, shutil, time
import numpy as np

def do_resize(img,small_size):
  size = img.size
  if size[0]<size[1]:
    new_size = ( small_size, int(small_size*size[1]/size[0]) )
  else:
    new_size = ( int(small_size*size[0]/size[1]), small_size )
  new_img=img.resize(new_size)
  #print "resize to %d,%d" % new_size
  return new_img

def do_crop(img,crop,position):
  if img.size[0] < crop[0]:
    raise Exception('img size[0] %d is smaller than crop[0]: %d' % (img[0],crop[0]))
  if img.size[1] < crop[1]:
    raise Exception('img size[1] %d is smaller than crop[1]: %d' % (img[1],crop[1]))
  if position == 'left_top':
    left=0
    upper=0
  if position == 'left_bottom':
    left=0
    upper=img.size[1]-crop[1]
  if position == 'right_top':
    left=img.size[0]-crop[0]
    upper=0
  if position == 'right_bottom':
    left=img.size[0]-crop[0]
    upper=img.size[1]-crop[1]
  if position == 'center':
    left=(img.size[0]-crop[0])/2
    upper=(img.size[1]-crop[1])/2

  box =(left,upper,left+crop[0],upper+crop[1])
  new_img = img.crop(box)
  #print "crop to box %d,%d,%d,%d" % box
  return new_img

def do_flip(img):
  new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
  return new_img

def load_img(path, grayscale=False):
  from PIL import Image
  img = Image.open(path)
  if grayscale:
      img = img.convert('L')
  else:  # Ensure 3 channel even when loaded image is grayscale
      img = img.convert('RGB')
  return img
  
def process_img(
            img,
            small_size,
            size,
            is_aug
                     ):
    im = load_img(img)
    im = do_resize(im,small_size)
    dataArray = []

    if is_aug:
        positions=["left_top","left_bottom","right_top","right_bottom","center"]
    else:
        positions=["center"]
    for position in positions:
        newIm=do_crop(im,size,position)
        assert newIm.size==size
        pix = np.array(newIm.convert("RGB"))
        dataArray.append(pix.transpose(2,0,1))
        if is_aug:
            newIm=do_flip(newIm)
            pix = np.array(newIm.convert("RGB"))
            dataArray.append(pix.transpose(2,0,1))

    return dataArray

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
