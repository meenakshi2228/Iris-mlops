import joblib

model = joblib.load("models/iris_model.joblib")

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

flower_names = ["Setosa", "Versicolor", "Virginica"]

print("Predicted flower:", flower_names[prediction[0]])