# ReptilePredictor

ReptilePredictor is a web application that can predict if a reptile image uploaded by the user shows a snake, a lizard or a crocodile. The backend of this web app is a transfer learning `ResNet50` model implemented in Keras. The frontend, on the other hand, was made using Dash.

## Installation

To run the website locally, please perform the following steps.

1. Clone this repository into a folder in your machine

```
git clone https://github.com/meraldoantonio/ReptilePredictor.git
cd ReptilePredictor
```

2. Create a virtual environment and install the dependencies within the virtual environment
```
conda create -n reptilepredictorenv python=3.6
conda activate reptilepredictorenv
pip install -r requirements.txt
```

3. Run the application. The application will download and serve the model.
```
python main.py
```


## Usage

1. Upload a jpeg image of either a snake, a crocodile or a lizard in the field provided. **Your image has to be in jpeg format!**

2. The probability assigned to each class will be shown as a bar chart.

3. Hover over the bar chart to load additional information about the probability of each class.

![image](https://drive.google.com/uc?export=view&id=1PIhqbSBQinE41lcA5e4YRK95btfuP_Iw)
