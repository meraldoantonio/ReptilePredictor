"""This is a simple Dash application where user can upload a picture of reptile and a backend Keras model will predict whether it is a snake, a crocodile or a lizard."""
import os
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from keras.models import load_model
import plotly.graph_objs as go
from PIL import Image
from utils import save_file, encode_image, preprocess_PIL_img, save_mpl_comparison_img 
import gdown

app = dash.Dash(__name__)
app.config['suppress_callback_exceptions'] = True

print("\n...Checking if you have downloaded the model, please wait...\n")
CWD = os.getcwd()
MODEL_DIR_NAME = "assets"
MODEL_NAME = "best.h5"
MODEL_DIR = os.path.join(CWD, MODEL_DIR_NAME)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

if not os.path.exists(MODEL_PATH):
    print(f"No model detected!")
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
        print(f"Created a folder `{MODEL_DIR_NAME}` in current working directory that will host the model.")
    MODEL_AVAILABLE = False
else:
    MODEL_AVAILABLE = True

if not MODEL_AVAILABLE:
    print("\n...Downloading model from GDrive (takes ~30 s)...\n")
    MODEL_URL = "https://drive.google.com/uc?id=1wPjd_G9eD6NkhzoNOS8lA8GE0H5s2KHH"
    gdown.download(MODEL_URL, MODEL_PATH, quiet = False)
    print(f"\n..Trained ResNet50 model successfully downloaded to {MODEL_PATH}...\n")

print("\n...Loading model, please wait...\n")
TRAINED_MODEL = load_model(MODEL_PATH)
TRAINED_MODEL._make_predict_function()

# Layout
app.layout = html.Div(children=[
    html.H1(children="Reptile Predictor", style={'textAlign': 'center'}),
    html.H2(children="Please upload a jpeg image of either a snake, a crocodile or a lizard:",
            style={'textAlign': 'center'}),
    dcc.Upload(id='upload-image',
               children=[html.Button('Upload image')],
               style={'textAlign': 'center'}),
    html.Div(id='img-div', style={'textAlign': 'center'}),
    html.P(),
    html.Div(id='prediction-div', style={'textAlign': 'center'}),
    html.Div(id='content-div', style={'textAlign': 'center'}) #'width': '22%'
])

# Callbacks
@app.callback(Output('img-div', 'children'),
              [Input('upload-image', 'contents')])
def display_image(contents):
    """Display uploaded image"""
    # Once an image is uploaded:
    if contents is not None:

        # Save it as jpg
        JPG_PATH = "assets/image.jpg"
        save_file(target_file_path=JPG_PATH, content=contents)

        # Open it as a PIL image and preprocess it to form numpy tensor image
        img_PIL = Image.open(JPG_PATH)
        preprocessed_img_np, preprocessed_img_np_batch = preprocess_PIL_img(img_PIL)

        # Create a comparison image that juxtaposes the original and preprocessed images
        FINAL_PNG_PATH = "assets/image.png"
        save_mpl_comparison_img(img_PIL, preprocessed_img_np, FINAL_PNG_PATH)

        # Show the comparison image
        return [html.Img(src=encode_image(FINAL_PNG_PATH)), html.H5("Classifying, please wait...")]

@app.callback(Output('prediction-div', 'children'),
              [Input('upload-image', 'contents')])
def display_prediction(contents):
    """Display prediction bar charts"""

    # Once the image is uploaded:
    if contents is not None:

        # Save it as jpg
        JPG_PATH = "assets/image.jpg"
        save_file(target_file_path=JPG_PATH, content=contents)

        # Open it as a PIL image and preprocess it to form numpy tensor image
        img_PIL = Image.open(JPG_PATH)
        preprocessed_img_np, preprocessed_img_np_batch = preprocess_PIL_img(img_PIL)

        # Use the proprocessed numpy tensor image to perform prediction
        prediction = TRAINED_MODEL.predict(preprocessed_img_np_batch)

        # Show the bar chart that contains the prediction
        return [html.H5("Here is our prediction:"),
                dcc.Graph(id="prediction-bar-chart",
                          figure=go.Figure(data=[go.Bar(x=["Crocodile", "Lizard", "Snake"],
                                                        y=prediction[0])],
                                           layout=go.Layout(xaxis=dict(title='Class',
                                                                       tickfont=dict(family='Old Standard TT, serif',
                                                                                     size=15,
                                                                                     color='black')),
                                                            yaxis={'title': 'Probability'},
                                                            autosize=True,
                                                            width=600,
                                                            height=600)),
                          style={'textAlign': 'center', "align-items": "center", 'display': 'inline-block', 'margin': 'auto'}),
                html.H5("Hover over any of the bars above to get more information, which will be displayed below.")
                ]

@app.callback(Output('content-div', 'children'),
              [Input('prediction-bar-chart', 'hoverData')])
def display_content(contents):
    """Display text underneath bar charts"""
    if contents is not None:
        animal_class = contents["points"][0]["x"]
        probability = contents["points"][0]["y"]*100
        H4_animal_class_information = html.H4(animal_class, style={'textAlign': 'center'})
        H4_probability_information = html.H4(f"Probability based on your image={probability:.2f}%",
                                             style={'textAlign': 'center'})
        return [H4_animal_class_information, H4_probability_information]

if __name__ == '__main__':
    app.run_server()
