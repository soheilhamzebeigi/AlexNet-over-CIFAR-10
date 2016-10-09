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
'''
To use this script, user should install these dependencies: flask pillow and protobuf
'''

from multiprocessing import Process, Queue
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import sys, os
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import time
import thread
import run

__all__ = []
__version__ = 0.1
__date__ = '2016-07-20'
__updated__ = '2016-10-08'
__shortdesc__ = '''
welcome to rafiki
'''

sys.path.append(os.getcwd())
app = Flask(__name__)
debug = False
top_k_ = 5
data = []


def success(data=""):
    res = dict(result="success", data=data)
    return jsonify(res)


def failure(message):
    res = dict(result="message", message=message)
    return jsonify(res)


def main(argv=None):
    '''Command line options'''
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version,
                                                     program_build_date)
    program_shortdesc = __shortdesc__
    program_license = '''%s

  Created by dbsystem group on %s.
  Copyright 2016 NUS School of Computing. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(
            description=program_license,
            formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument(
            "-p",
            "--port",
            dest="port",
            default=9999,
            help="the port to listen to, default is 5000")
        parser.add_argument(
            "-param",
            "--parameter",
            dest="parameter",
            help="the parameter file path to be loaded")
        parser.add_argument(
            "-D",
            "--debug",
            dest="debug",
            action="store_true",
            help="whether need to debug")
        parser.add_argument(
            "-R",
            "--reload",
            dest="reload_data",
            action="store_true",
            help="whether need to reload data")
        parser.add_argument(
            "-C",
            "--cpu",
            dest="use_cpu",
            action="store_true",
            help="Using cpu or not, default is using gpu")
        parser.add_argument(
            '-V',
            '--version',
            action='version',
            version=program_version_message)

        # Process arguments
        args = parser.parse_args()

        port = args.port
        parameter_file = args.parameter
        need_reload = args.reload_data
        use_cpu = args.use_cpu
        debug = args.debug

        global queue
        queue = Queue()

        p = Process(
            target=run.model_run,
            args=(port, parameter_file, need_reload, use_cpu, debug, queue))
        p.start()

        app.debug = debug
        app.run(host='0.0.0.0', port=port)

    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception, e:
        if debug:
            traceback.print_exc()
            raise (e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + str(e) + "\n")
        sys.stderr.write(indent + "  for help use --help \n\n")
        return 2


@app.route("/")
@cross_origin()
def index():
    return "Hello, this is a rafiki server. You can create, start, stop, \
    delete a job and getData from it!\n"


def getDataFromQueue():
    print 'hello queue size: ', queue.qsize()
    while not queue.empty():
        d = queue.get()
        data.append(d)


@app.route('/getAllData')
@cross_origin()
def getAllData():
    getDataFromQueue()
    return success(data)


@app.route('/getTopKData')
@cross_origin()
def getTopKData():
    k = request.args.get("k", top_k_)
    k = int(k)
    getDataFromQueue()
    return success(data[-k:])


if __name__ == '__main__':
    main()
