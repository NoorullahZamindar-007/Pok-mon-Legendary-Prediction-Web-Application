       
  
# Pokémon Legendary Prediction Web Application  

This project is a machine learning–powered web application that predicts whether a Pokémon is Legendary or Not Legendary based on its attributes. The application integrates a trained Random Forest classification model with a Flask-based web interface to provide real-time predictions and model insights.

## Overview   
 
The Pokémon Legendary Prediction Web Application allows users to input Pokémon characteristics such as base stats, generation, types, physical attributes, and special properties. Based on these inputs, the system predicts the Legendary status and provides the probability associated with the prediction. 
  
The project demonstrates the complete machine learning lifecycle, including feature engineering, model training, and deployment as a web application.
 
## Features  
    
- Predicts Legendary or Not Legendary Pokémon  
- Displays probability score for Legendary classification     
- Supports multiple Pokémon attributes including base stats and types 
- Interactive dashboard showing feature importance
- Clean, responsive, and user-friendly interface
- Real-time prediction using a trained machine learning model

## Technology Stack

- Python
- Flask
- Scikit-learn
- NumPy
- Bootstrap
- Chart.js

## Machine Learning Model

The prediction model is a Random Forest Classifier trained on Pokémon dataset features, including:

- Base stats (HP, Attack, Defense, Special Attack, Special Defense, Speed)
- Pokémon generation
- Pokémon types (one-hot encoded)
- Pokémon color
- Physical attributes (height and weight)
- Catch rate
- Special properties such as Mega Evolution and Gender availability

The model outputs both the predicted class and the probability of a Pokémon being Legendary.

## Project Structure

```

pokemon-flask-app/
│── app.py
│── pokemon_model.pickle
│── requirements.txt
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── dashboard.html
│
└── static/
├── css/
│   └── style.css
└── js/
└── app.js

```

## Installation and Setup

1. Clone the repository:
```

git clone [https://github.com/

```

2. Navigate to the project directory:
```

cd pokemon-legendary-prediction

```

3. Install required dependencies:
```

pip install -r requirements.txt

```

4. Run the Flask application:
```

python app.py

```

5. Open your browser and visit:
```

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

```

## Usage

- Enter Pokémon attributes in the input form
- Select Pokémon type(s) and color
- Submit the form to get the Legendary prediction
- View prediction probability and input summary
- Access the dashboard to analyze feature importance

## Use Case

This project demonstrates how machine learning models can be deployed into production-like environments using Flask. It is suitable for learning purposes, portfolio presentation, and as a reference for deploying classification models as web applications.

## Author

Noorullah Zamindar  
Machine Learning and AI Engineer
```

---

