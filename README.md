# Intro to Machine Learning - TensorFlow Project

Project code for Udacity's Intro to Machine Learning with TensorFlow Nanodegree program. In this project, you will first develop code for an image classifier built with TensorFlow, then you will convert it into a command line application.

In order to complete this project, you will need to use the GPU enabled workspaces within the classroom.  The files are all available here for your convenience, but running on your local CPU will likely not work well.

You should also only enable the GPU when you need it. If you are not using the GPU, please disable it so you do not run out of time!

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github.  If you would like the data for this project, you will want download it from the workspace in the classroom.  Though actually completing the project is likely not possible on your local unless you have a GPU.  You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.

## Part 1: Developing an Image Classifier with Deep Learning

In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with TensorFlow. We'll provide some tips and guide you, but for the most part the code is left up to you.

Remember that your code should be your own, please do not plagiarize.

This notebook will be required as part of the project submission. After you finish it, make sure you download it as an HTML file and include it with the files you write in the next part of the project.

We've provided you a workspace with a GPU for working on this project.

If you are using the workspace, be aware that saving large files can create issues with backing up your work. You'll be saving a model checkpoint in Part 1 of this project which can be multiple GBs in size if you use a large classifier network. Dense networks can get large very fast since you are creating N x M weight matrices for each new layer. In general, it's better to avoid wide layers and instead use more hidden layers, this will save a lot of space. Keep an eye on the size of the checkpoint you create. You can open a terminal and enter `ls -lh` to see the sizes of the files. If your checkpoint is greater than 1 GB, reduce the size of your classifier network and re-save the checkpoint.

## Part 2 - Building the Command Line Application

Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a Python script that run from the command line. For testing, you should use the saved Keras model you saved in the first part.

### Specifications

The project submission must include a `predict.py` file that uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a module just for utility functions like preprocessing images. Make sure to include all files necessary to run the `predict.py` file in your submission.

The `predict.py` module should predict the top flower names from an image along with their corresponding probabilities.

#### Basic usage:

```
$ python predict.py /path/to/image saved_model
```

#### Options:

- --top_k : Return the top KK most likely classes:
```
$ python predict.py /path/to/image saved_model --top_k KK
```

- --category_names : Path to a JSON file mapping labels to flower names:
```
$ python predict.py /path/to/image saved_model --category_names map.json
```

The best way to get the command line input into the scripts is with the `argparse` module in the standard library.

### Examples:

For the following examples, we assume we have a file called `orchid.jpg` in a folder named `/test_images/` that contains the image of a flower. We also assume that we have a Keras model saved in a file named `my_model.h5`.

#### Basic usage:
```
$ python predict.py ./test_images/orchid.jpg my_model.h5
```

#### Options:

- Return the top 3 most likely classes:
```
$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
```

- Use a `label_map.json` file to map labels to flower names:
```
$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
```

### Images for Testing

In the workspace we have provided 4 images in the `./test_images/` folder for you to check your `prediction.py` module. The 4 images are:

- cautleya_spicata.jpg
- hard-leaved_pocket_orchid.jpg
- orange_dahlia.jpg
- wild_pansy.jpg
