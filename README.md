# Train AlexNet over CIFAR-10

This example provides the training and serving scripts for [AlexNet](https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg) over [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) data.
The best validation accuracy (without data augmentation) we achieved was about 82%.


## SINGA version

Note that all examples should clearly specify the SINGA version against which the scripts are tested. The format is Apache SINGA-<VERSION>-<COMMITID>. For example,
All scripts have been tested against [Apache SINGA-v1.0.0-fac3af9](https://github.com/apache/incubator-singa/commit/fac3af94990e4c9977f82a11ae85f89de9dbb463).


## Folder layout

The folder structure for an example is as follows where README.md is required and other files are optional.

* README.md. Every example **should have** a README.md file for the model description, SINGA version and running instructions.
* train.py. The training script. Users should be able to run it directly by `python train.py`. It is optional if the model is shared only for prediction or serving tasks.
* serve.py. The serving script. It is typically used in the cloud mode, where users can submit the query via the web front end provided by Rafiki. If the local mode is enabled, it should accepts command line input. It is optional if the model is shared only for training tasks.
* model.py. It has the functions for creating the neural net. It could be merged into train.py and serve.py, hence are optional.
* data.py. This file includes functions for downloading and extracting data and parameters. These functions could be merged into the train.py and serve.py, hence are optional.
* index.html. This file is used for the serving task, which provides a web page for users to submit queries and get responses for the results. If is required for running the serving task in the cloud mode. If the model is shared only for training or running in the local mode, it is optional.
* requirements.txt. For specifying the python libraries used by users' code. It is optional if no third-party libs are used.

Some models may have other files and scripts. Typically, it is not recommended to put large files (e.g. >10MB) into this folder as it would be slow to clone the gist repo.

## Instructions

### Local mode
To run the scripts on your local computer, you need to install SINGA.
Please refer to the [installation page](http://singa.apache.org/en/docs/installation.html) for detailed instructions.

#### Training
The training program could be started by

        python train.py

By default, the training is conducted on a GPU card, to use CPU for training (very slow), run

        python train.py --use_cpu

The model parameters would be dumped periodically, into `model-<epoch ID>` and the last one is `model`.

#### Serving

This example does not have the serving script for local mode. To simulate the local mode, you can start the prediction script and use curl to pass the query image.

        python serve.py &
        curl -i -F image=@image1.jpg http://localhost:9999/api
        curl -i -F image=@image2.jpg http://localhost:9999/api
        curl -i -F image=@image3.jpg http://localhost:9999/api

The above commands start the serving program using the model trained for Alexnet as a daemon,
and then submit three queries (image1.jpg, image2.jpg, image3.jpg) to the port (the default port is 9999).
To use other port, please add `-p PORT_NUMBER` to the running command.
If you run the serving task after finishing the training task,
then the model parameters from `model` would be used.
Otherwise, it would use the one downloaded using data.py.


### Cloud mode

To run the scripts on the Rafiki platform, you don't need to install SINGA. But you need to add the dependent libs introduced by your code into the requirement.txt file.

#### Adding model

The Rafiki front-end provides a web page for users to import gist repos directly.
Users just specify the HTTPS (NOT the git web URL) clone link and click `load` to import a repo.

#### Training

The Rafiki font-end has a Job view for adding a new training job. Users need to configure the job type as 'training',
select the model (i.e. the repo added in the above step) and its version.
With these fields configured, the job could be started by clicking the `start` button.
Afterwards, the users would be redirected to the monitoring view.
Note that it may take sometime to download the data for the first time.
The Rafiki backend would run `python train.py` in the backend.

#### Serving

The serving job is similar to the training job except the job type is 'serving'.
The Rafiki backend would run `python serve.py`. Users can jump to the serving view rendered using the `index.html` from the gist repo.
Note that it may take sometime to download the data for the first time.
