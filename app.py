import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset from CSV
data = pd.read_csv("iris.csv")

# Separate features and target variable
X = data.drop("species", axis=1)
y = data["species"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Create a Streamlit web application
st.title("Iris Flower Classification")
st.write("Enter the measurements below to predict the Iris species:")

# Collect input from the user
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Make a prediction on the user input
user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = classifier.predict(user_input)

# Display the predicted species
st.write("Predicted Species:", prediction[0])

# Calculate the accuracy of the classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)
