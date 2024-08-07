import cv2
import sys
import numpy as np
from tensorflow.keras import models
import tkinter as tk
from tkinter import filedialog

MODEL_PATH = "tf-cnn-model.h5"

def predict_digit(image_path):
    # load model
    model = models.load_model(MODEL_PATH)
    print("[INFO] Loaded model from disk.")

    image = cv2.imread(image_path, 0)      
    image1 = cv2.resize(image, (28,28))    # For cv2.imshow: dimensions should be 28x28
    image2 = image1.reshape(1,28,28,1)

    cv2.imshow('digit', image1 )
    pred = np.argmax(model.predict(image2), axis=-1)
    return pred[0]    

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(self)
        self.load_button["text"] = "Load Image"
        self.load_button["command"] = self.load_image
        self.load_button.pack(side="top")

        self.predict_button = tk.Button(self)
        self.predict_button["text"] = "Predict Digit"
        self.predict_button["command"] = self.predict_digit
        self.predict_button.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

        self.image_path = ""

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        print(f"Loaded image: {self.image_path}")

    def predict_digit(self):
        if self.image_path:
            predicted_digit = predict_digit(self.image_path)
            print(f"Predicted Digit: {predicted_digit}")
        else:
            print("No image loaded")

root = tk.Tk()
app = Application(master=root)
app.mainloop()