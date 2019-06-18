FROM python:3

CMD sudo apt-get update && apt-get upgrade

# Install requirements
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt


# Flask app
COPY api /api
WORKDIR /api/

CMD ["gunicorn", "app:app", "-c", "config.py"]