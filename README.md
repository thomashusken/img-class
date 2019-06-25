# Image classification
This is a webapp for image classification using the Inception network. 

## Run

The easiest way to run this application is as a docker container. First, clone the repository:

```git clone git@github.com:thomashusken/img-class.git```

Navigate to the repository, then build the image with: 

```docker build -t imgclass .```

Run the image as a container:

```docker run -d -p 5000:5000 imgclass```

Happy classifying!