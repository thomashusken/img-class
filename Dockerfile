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
ENV MODEL_PATH="data/"
CMD ["python", "import torchvision", "torch.utils.model_zoo.load_url($TORCH_MODEL_URL, $MODEL_PATH)"]

CMD ["gunicorn", "app:app", "-c", "config.py"]