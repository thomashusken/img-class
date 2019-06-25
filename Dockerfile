FROM python:3

CMD sudo apt-get update && apt-get upgrade


# Install requirements
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Flask app
COPY api /api
COPY data /api/data
WORKDIR /api/

# Download inception model
ENV TORCH_MODEL_URL="https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
ENV MODEL_PATH="data/inception_v3_google-1a9a5a14.pth"
RUN curl $TORCH_MODEL_URL --output $MODEL_PATH

#CMD ["python", "import torchvision", "import os", \
#   "torch.utils.model_zoo.load_url(os.environ['TORCH_MODEL_URL'], os.environ['MODEL_PATH'])"]

CMD ["gunicorn", "app:app", "-c", "config.py"]