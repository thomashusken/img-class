from flask import Flask, render_template, request, send_file
from flask_bootstrap import Bootstrap
from .etl import ImageParser
from .model import return_top_5
from torchvision import transforms
import io
import base64


def create_app():
    app = Flask(__name__)
    Bootstrap(app)
    return app


app = create_app()

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

        img_io = io.BytesIO()
        raw_image.save(img_io, 'PNG', quality=100)
        img_io.seek(0)
        return_image = base64.b64encode(img_io.getvalue()).decode('ascii')

        # return_image = serve_pil_image(raw_image)
        prediction = return_top_5(processed_image)
    return render_template('upload.html', return_image=return_image, prediction=prediction)

def serve_pil_image(pil_img):
    output = io.BytesIO()
    pil_img.convert('RGBA').save(output, format='PNG')
    output.seek(0, 0)
    output = base64.b64encode(output.getvalue())

    return send_file(output, mimetype='image/png', as_attachment=False)


@app.route('/about/')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run()
