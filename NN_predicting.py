# load and evaluate a saved model
#necessary imports
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def findmaxvalue(yHat): #finding largest yHat value to limit the oerfect relationship line
    max_value = 0
    for number in yHat:
        if (number > max_value):
            max_value = number
    return max_value

model = tf.keras.models.load_model("advertising_model") #load model
#or you could use: model = load_model("model_here.h5") to load models saved in h5 files
model.summary() #load dataset

dirName = filedialog.askopenfile(initialdir="/",title='Please select a file') #choosing which files to read/getting the file path
df = pd.read_csv(dirName) #reading csv

df.fillna(method ='ffill', inplace = True) #removing holes or nonvalues
df.dropna(inplace = True) #removing holes or nonvalues

x = df[["TV", "Newspaper", "Radio"]] #reading columns for x
y = df[["Sales"]] #reading column for y


yHat = model.predict(x) #predicting
xLimit = findmaxvalue(yHat) #finding max value for yhat for the line that won't be used

_, accuracy = model.evaluate(x, y) #this just evaluates the accuracy of the model
print("Accuracy: ", accuracy)

x1 = np.linspace(0, xLimit, 100) #creating the x range
plt.scatter(yHat, y, color = "b") #scattering yhat values
#plt.plot(x1, x1+0, "r", label = "Perfect Relationship") #plotting perfect relationship line that isn't used right now
plt.legend()
plt.xlabel("Y hat predicted values") #labels and titles 
plt.ylabel("Actual y values")
plt.title("Relationship between Y Hat values and Real Y values")
plt.show() #show the amazing graph