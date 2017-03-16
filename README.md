# Caption This!
This repository contains source code corresponding to our article ["Caption this, with TensorFlow!"]( https://www.oreilly.com/learning/caption-this-with-tensorflow).

# Docker
*Running with Docker is highly recommended*
You can find platform-specific installation instructions for Docker [here](https://docs.docker.com/engine/installation/#platform-support-matrix). Our iPython notebooks are compatible with TensorFlow 1.0.

After installing Docker, either use the Dockerfiles provided in `./dockerfiles/` to build a docker image with `docker build -t <image_name> ./dockerfiles/`, or (recommended for cpu users) pull a prebuilt image from our dockerhub with `docker pull mlatberkeley/showandtell`.

After building, starting, and attaching to the appropriate Docker container (details [here](https://docs.docker.com/engine/getstarted/step_one/#looking-for-troubleshooting-help)) run the provided jupyter notebooks with `jupyter notebook --ip 0.0.0.0` and follow the instructions on screen.

*Note*
If you are using Docker Toolbox as opposed to native Docker you will have to navigate to the Daemon IP adress (instead of 0.0.0.0) provided right after starting the Docker Quickstart Terminal (for us this was 192.168.99.100) in order to use jupyter.


# The Notebooks
There are three notebooks:
* `O'Reilly Training.ipynb` - Contains code to train a TensorFlow caption generator from a VGG16 word embedding as described in our article. *Note:* you must run this notebook's `train` method before running any of the other notebooks in order to generate a mapping between integers and our vocabulary's words that will be reused in the other notebooks.
* `O'Reilly Generate.ipynb` - Contains the same code as `O'Reilly Training.ipynb` except it introduces functionality to generate captions from an image embedding (as opposed to just being able to train on captions). Functions as a sanity check for the quality of captions we are generating.
* `O'Reilly Generate from image.ipynb` - Builds on the previous notebook, except instead of feeding an image embedding to our caption generation model, it first feeds an image to the VGG-16 Convolutional Neural Network to generate an image feature embedding. This gives us an end-to-end pipeline for going from an image to a caption.
 * In order to run the test notebook edit the image path in the ipynb (more details in the `.ipynb` itself).

# Additional Downloads:
In order to run the first two notebooks, you will need VGG-16 image embeddings for the Flickr-30K dataset. These image embeddings are available from our [Google Drive](https://drive.google.com/file/d/0B5o40yxdA9PqTnJuWGVkcFlqcG8/view?usp=sharing).

Additionally, you will need the corresponding captions for these images, which can also be downloaded from our [Google Drive](https://drive.google.com/file/d/0B2vTU3h54lTydXFjSVM5T2t4WmM/view?usp=sharing).

In order to run the `O'Reilly Generate from image.ipynb` notebook you will need to download a pretrained TensorFlow model for [VGG-16](https://drive.google.com/file/d/0B2vTU3h54lTyaDczbFhsZFpsUGs/view?usp=sharing) generated from the original Caffe model from the VGG-16 paper. 

Place all of these downloads in the `./data/` folder.
