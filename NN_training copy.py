#training and then saving a NN that uses the Advertising csv file
#necessary imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd
from tkinter import filedialog

dirName = filedialog.askopenfile(initialdir="/",title='Please select a file') #choosing which files to read/getting the file path

df = pd.read_csv(dirName) #reading csv

df.fillna(method ='ffill', inplace = True) #removing holes or nonvalues
df.dropna(inplace = True) #removing holes or nonvalues

x = df[["TV", "Newspaper", "Radio"]] #reading columns for x
y = df[["Sales"]] #reading column for y

x = np.array(x) #array for x values
y = np.array(y) #array for y values

model = Sequential([Dense(units = x.shape[1], activation = "relu"), #the actual actitecture has the be improved on. I made this as a base template
                    Dense(units = 100, activation = "relu"),
                    Dense(units = 50, activation = "relu"),
                    Dense(units = 25, activation = "relu"),
                    Dense(units = 1, activation = "linear")])


model.compile(optimizer = "Adam", loss = "mse", metrics = ["mean_squared_error"]) #compiling using the adam algorithm and the means squared error as the loss function
model.fit(x, y, epochs = 5000, batch_size = 128) #the model goes through until sets of 128 samples have updated the values 5000 times
model.summary()
print("Done Training")

_, accuracy = model.evaluate(x, y) #this just evaluates the accuracy of the model
print("Accuracy: ", accuracy)

model.save("advertising_model")
#or you could save the model in a h5 file using model.save("advertising_model.h5")
print("model saved")