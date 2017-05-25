# Caption This!

This repository contains source code corresponding to our article ["Caption this, with TensorFlow!"]( https://www.oreilly.com/learning/caption-this-with-tensorflow) 


# Git Basics
1. Go to your home directory by opening your terminal and entering `cd ~`

2. Clone the repository by entering

    ```
    git clone https://github.com/mlberkeley/oreilly-captions.git
    ```

# Docker (highly recommended)
Install Docker using the platform-specific installation instructions for Docker [here](https://docs.docker.com/engine/installation/#platform-support-matrix). Our iPython notebooks are compatible with TensorFlow 1.0.

### Option A: Use our pre-built Docker image from Docker Hub

3. After installing Docker, pull a prebuilt image from our Docker Hub by entering:

    ```
    docker pull mlatberkeley/showandtell
    ```

    You will need a Docker Hub account in order to pull the image (get one [here](https://hub.docker.com/)). If it's your first time pulling a Docker image from Docker Hub you will need to login to your Docker Hub account from your terminal with `docker login`, and follow the username and password prompt.

4. To run the pulled image (after cloning and downloading the repository) enter

    ```
    docker run -it -p 8888:8888 -v <path to repo>:/root mlatberkeley/showandtell
    ```

    where `<path to repo>` should be the __absolute path__ to your cloned repository. If you followed our **Git Basics** section the path should be `<path to your home directory>/oreilly-captions`.

5. After building, starting, and attaching to the appropriate Docker container, run the provided Jupyter notebooks by entering

    ```
    jupyter notebook --ip 0.0.0.0
    ```

    and navigate to [http://0.0.0.0:8888](http://0.0.0.0:8888) in your browser.

### Option B: Download and build your own Docker image from our GitHub repo
If you want to build a GPU or CPU-based Docker image of your own, you can use the Dockerfiles provided in the `/dockerfiles/` subdirectory of our GitHub repo.

3. After cloning the repo to your machine, enter

    ```
    docker build -t showandtell_<image_type> -f ./dockerfiles/Dockerfile.<image_type> ./dockerfiles/
    ```

    where `<image_type>` is either `gpu` or `cpu`. (Note that, in order to run these files on your GPU, you'll need to have a compatible GPU, with drivers installed and configured properly [as described in TensorFlow's documentation](https://www.tensorflow.org/install/).)

4. Run the Docker image by entering

    ```
    docker run -it -p 8888:8888 -v <path to repo>:/root showandtell_<image_type>
    ```

    where `<image_type>` is either `gpu` or `cpu`, depending on the image you built in the last step.

5. After building, starting, and attaching to the appropriate Docker container, run the provided Jupyter notebooks by entering

    ```
    jupyter notebook --ip 0.0.0.0
    ```

    and navigate to [http://0.0.0.0:8888](http://0.0.0.0:8888) in your browser.

**Note**
If you are using Docker Toolbox as opposed to native Docker you will have to navigate to the Daemon IP adress (instead of 0.0.0.0) provided right after starting the Docker Quickstart Terminal (for us this was 192.168.99.100) in order to use Jupyter.

### Debugging docker
If you receive an error of the form:

```
WARNING: Error loading config file:/home/rp/.docker/config.json - stat /home/rp/.docker/config.json: permission denied
Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.26/images/json: dial unix /var/run/docker.sock: connect: permission denied
```

It's most likely because you installed Docker using sudo permissions with a packet manager such as `brew` or `apt-get`. To solve this `permission denied` simply run docker with `sudo` (ie. run `docker` commands with `sudo docker <command and options>` instead of just `docker <command and options>`).

# The Notebooks
There are three notebooks:
* `1. O'Reilly Training.ipynb` - Contains code to train a TensorFlow caption generator from a VGG16 word embedding as described in our article. *Note:* you must run this notebook's `train` method before running any of the other notebooks in order to generate a mapping between integers and our vocabulary's words that will be reused in the other notebooks.
* `2. O'Reilly Generate.ipynb` - Contains the same code as `1. O'Reilly Training.ipynb` except it introduces functionality to generate captions from an image embedding (as opposed to just being able to train on captions). Functions as a sanity check for the quality of captions we are generating.
* `3. O'Reilly Generate from image.ipynb` - Builds on the previous notebook, except instead of feeding an image embedding to our caption generation model, it first feeds an image to the VGG-16 Convolutional Neural Network to generate an image feature embedding. This gives us an end-to-end pipeline for going from an image to a caption.
 * In order to run the test notebook edit the image path in the ipynb (more details in the `.ipynb` itself).

# Additional Downloads:
In order to run the first two notebooks, you will need VGG-16 image embeddings for the Flickr-30K dataset. These image embeddings are available from our [Google Drive](https://drive.google.com/file/d/0B5o40yxdA9PqTnJuWGVkcFlqcG8/view?usp=sharing).

Additionally, you will need the corresponding captions for these images (`results_20130124.token`), which can also be downloaded from our [Google Drive](https://drive.google.com/file/d/0B2vTU3h54lTydXFjSVM5T2t4WmM/view?usp=sharing).

In order to run the `3. O'Reilly Generate from image.ipynb` notebook you will need to download a pretrained TensorFlow model for [VGG-16](https://drive.google.com/file/d/0B2vTU3h54lTyaDczbFhsZFpsUGs/view?usp=sharing) generated from the original Caffe model from the VGG-16 paper.

Place all of these downloads in the `./data/` directory.
