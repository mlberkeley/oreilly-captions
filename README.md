# Caption This!
This repo contains material for our TF caption generation from images article.

# Docker
*Running with Docker is highly recommended*
Our Notebookes were written with the release of TF 1.0 which has undergone a few changes in function definitions from TF 0.12. To run the code in this repo we highly recommend using and installing [Docker](https://docs.docker.com/engine/installation/#platform-support-matrix) for your respective system.

After installing docker, either use the Dockerfiles provided in `./dockerfiles/` to build a docker image with `docker build -t <image_name> ./dockerfiles/`. However, we also recommend pulling a prebuilt image from our dockerhub with `docker pull mlatberkeley/showandtell`.

After attaching to the appropriate docker container (details [here](https://docs.docker.com/engine/getstarted/step_one/#looking-for-troubleshooting-help)) run the provided jupyter notebooks with `jupyter notebook --ip 0.0.0.0` and follow the instructions on screen.

*Note*
If you are using Docker Toolbox as opposed to native Docker you will have to navigate to the Daemon IP adress (instead of 0.0.0.0) provided right after starting the Docker Quickstart Terminal (for us this was 192.168.99.100) in order to use jupyter.


# The Notebooks
There are three notebooks:
* `O'Reilly Training.ipynb` - Contains code to train a tensorflow caption generator from a VGG16 word embedding.
* `O'Reilly Generate.ipynb` - Contains the same code as `O'Reilly Training.ipynb` except it introduces functionality to generate captions at test time with beam search.
* `O'Reilly Generate from image.ipynb` - Builds on the previous notebook, except instead of feeding an image embedding to our caption generation model, it first feeds an image to VGG-16 to generate the image embedding. Thus, giving us an end-to-end pipeline for going from an image to a Caption.
 * In order to run the test notebook edit the image path in the ipynb.

# Additional Downloads:
In order to run the first two notebooks you will need VGG-16 image embeddings for the Flikr-30K dataset. These image embeddings are available from our [google drive](https://drive.google.com/file/d/0B5o40yxdA9PqTnJuWGVkcFlqcG8/view?usp=sharing).

Additionally, you will need the corresponding captions for these images, which can also be downloaded from our [google drive](https://drive.google.com/file/d/0B2vTU3h54lTydXFjSVM5T2t4WmM/view?usp=sharing)

In order to run the `O'Reilly Generate from image.ipynb` notebook you will need to download a pretrained tensorflow model for [VGG-16](https://drive.google.com/file/d/0B2vTU3h54lTyaDczbFhsZFpsUGs/view?usp=sharing) generated from the original Caffe model from the VGG-16 paper. We've linked to ry's `vgg-16.tfmodel` as it is the original opensource port of vgg16 to tensorflow. 

Place all of these downloads in the `./data/` folder.
