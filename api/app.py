from flask import Flask, render_template
from flask_bootstrap import Bootstrap

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

@app.route('/upload/')
def upload():
    return render_template('upload.html')

@app.route('/about/')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run()
