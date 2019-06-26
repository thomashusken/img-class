from flask import Flask, render_template, request, send_file
import json
from flask_bootstrap import Bootstrap
from etl import ImageParser
from model import return_top_5
from torchvision import transforms, models
import torch
import io
import base64


def create_app():
    app = Flask(__name__)
    Bootstrap(app)
    return app

# Create app
app = create_app()


# Initialise model and load weights from file
inception = models.inception_v3()
inception.load_state_dict(torch.load("data/inception_v3_google-1a9a5a14.pth"))
inception.eval()

#load imagenet classes
class_idx = json.load(open('data/imagenet_class_index.json'))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


@app.route('/test')
def test():
    return 'test'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    prediction = None
    return_image = None
    if request.method == 'POST':
        data = request.files['data']
        parser = ImageParser(data, transform=transforms.Compose([
         transforms.Resize((299, 299)),
         transforms.CenterCrop(299),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

        raw_image, processed_image = parser.load_image()
        return_image = serve_pil_image(raw_image)
        prediction_dict = return_top_5(processed_image, inception, idx2label)
        prediction = json.dumps(prediction_dict, sort_keys=False)
    return render_template('upload.html', return_image=return_image, prediction=prediction)

def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG', quality=100)
    img_io.seek(0)
    return_image = base64.b64encode(img_io.getvalue()).decode('ascii')

    return return_image


@app.route('/about/')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run()
