# Caption This!
This repo contains material for our TF caption generation from images article.

# The Notebooks
There are three notebooks:
* `O'Reilly Training.ipynb` - Contains code to train a tensorflow caption generator from a VGG16 word embedding.
* `O'Reilly Generate.ipynb` - Contains the same code as `O'Reilly Training.ipynb` except it introduces functionality to generate captions at test time with beam search.
* `O'Reilly Generate from image.ipynb` - Builds on the previous notebook, except instead of feeding an image embedding to our caption generation model, it first feeds an image to VGG-16 to generate the image embedding. Thus, giving us an end-to-end pipeline for going from an image to a Caption.
 * In order to run the test notebook edit the image path in the ipynb.

# Additional Downloads:
In order to run the first two notebooks you will need VGG-16 image embeddings for the Flikr-30K dataset. These image embeddings are available from our [google drive](https://drive.google.com/file/d/0B5o40yxdA9PqTnJuWGVkcFlqcG8/view?usp=sharing).

In order to run the `O'Reilly Generate from image.ipynb` notebook you will need to download a pretrained tensorflow model for [VGG-16](https://github.com/ry/tensorflow-vgg16) generated from the original Caffe model from the VGG-16 paper. We've linked to ry's `vgg-16.tfmodel` as it is the original opensource port of vgg16 to tensorflow. 

Place both of these downloads in the `./data/` folder.
