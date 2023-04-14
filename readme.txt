Documentation for Rec-App:

The Rec-App is a web-based application that provides personalized recommendations based on user input. The app is built on Flask, a Python-based web framework. It uses several machine learning libraries such as TensorFlow, Keras, and scikit-learn to train a model for predicting user input and providing personalized recommendations.

The main file in the app is app.py, which contains the following code:

Importing libraries:
The first step is to import the necessary libraries such as Flask, TensorFlow, Keras, NumPy, Pandas, etc.

Data Preprocessing:
The app uses the input data from the users to train a model for predicting user input and providing personalized recommendations. The input data is stored in intents.json file. The app reads this file and preprocesses the input data by removing punctuations, converting text to lowercase, tokenizing, and padding.

Model creation:
After preprocessing, the app creates a model using TensorFlow and Keras libraries. The model is a sequential model with several layers, including an embedding layer, LSTM layer, and dense layer. The model is trained using the preprocessed input data.

Serving the model:
Once the model is trained, it is ready to be served. The app uses Flask to serve the model on the web. The user input is passed to the model, and the model returns a prediction. The prediction is then used to provide personalized recommendations.

User Interface:
The app provides a user interface using HTML and CSS. The user interface includes a text input field where the user can input their query. The app provides personalized recommendations based on the user input. It also provides a link to the Lazada website with a search query based on the user input.

Overall, the Rec-App is a useful application that provides personalized recommendations based on user input. The app is built on Flask and uses several machine learning libraries such as TensorFlow, Keras, and scikit-learn to provide personalized recommendations.