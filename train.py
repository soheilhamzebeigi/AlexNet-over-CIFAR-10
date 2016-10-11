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

from multiprocessing import Process, Queue
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import sys, os, traceback

import flaskserver
import model
from trainer import Trainer

sys.path.append(os.getcwd())


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

        # start monitor server
        # use multiprocessing to transfer training status information
        queue = Queue()
        p = Process(target=flaskserver.start_monitor, args=(port, queue))
        p.start()

        # start to train
        m = model.create(use_cpu)
        trainer = Trainer(m, use_cpu, queue)
        trainer.initialize(parameter_file)
        trainer.train()

        p.terminate()
    except SystemExit:
        return
    except:
        #p.terminate()
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2


if __name__ == '__main__':
    main()
