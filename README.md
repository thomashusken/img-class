# Image classification
This is a webapp for image classification using the Inception network. It consists of a Flask
Bootstrap webapp, using PyTorch for the classification of images.

## How to run
There are currently two ways to run this application, cloning via git and building the docker image from the repo, 
or pulling the image from docker hub.

### Clone and build

First, clone the repository:

```git clone git@github.com:thomashusken/img-class.git```

Navigate to the repository, then build the image with: 

```docker build -t imgclass .```

### Pull from docker hub

To pull the image from docker hub, run

```docker pull cptslow/imgclass```

### Starting the container

Starting a container from this image is now straightforward:

```docker run -d -p 5000:5000 imgclass```


Happy classifying!