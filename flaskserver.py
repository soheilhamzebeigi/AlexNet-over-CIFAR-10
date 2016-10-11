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

from multiprocessing import Process
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

app = Flask(__name__)
top_k_ = 5

def success(data=""):
    res = dict(result="success", data=data)
    return jsonify(res)

def failure(message):
    res = dict(result="message", message=message)
    return jsonify(res)

def start_monitor(port,queue):
    global queue_,data_,type_
    queue_=queue
    data_=[]
    type_="monitor"
    app.run(host='0.0.0.0', port=port)
    return

def start_serve(port,service):
    global type_,service_
    service_=service
    type_="serve"
    app.run(host='0.0.0.0', port=port)
    return

def getDataFromQueue():
    global queue_,data
    while not queue_.empty():
        d = queue_.get()
        data_.append(d)

@app.route("/")
@cross_origin()
def index():
    global type_
    print type_
    if type_== "monitor": 
        return "Hello,This is SINGA monitor http server"
    else:
        return send_from_directory(".","index.html",mimetype='text/html')
   
@app.route("/predict",methods=['POST'])
@cross_origin() 
def predict():
    global type_,service_
    if type_=="monitor":
        return failure("not available in monitor mode")
    if request.method == 'POST':
        try:
            print "test"
            response=service_.serve(request)
        except Exception as e:
            print str(e)
            return e
        return response

@app.route('/getAllData')
@cross_origin()
def getAllData():
    global data_,type_
    if type_=="serve":
        return failure("not available in serve mode")
    getDataFromQueue()
    return success(data_)

@app.route('/getTopKData')
@cross_origin()
def getTopKData():
    global data_,type_
    if type_=="serve":
        return failure("not available in serve mode")
    k = request.args.get("k", top_k_)
    try:
        k = int(k)
    except:
        return failure("k should be integer")
    getDataFromQueue()
    return success(data_[-k:])
