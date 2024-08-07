import cv2
import sys
import numpy as np
from tensorflow.keras import models
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Entry, Button
import PIL.Image, PIL.ImageTk

MODEL_PATH = "tf-cnn-model.h5"

def predict_digit(image_path):
    # load model
    custom_objects = {'SparseCategoricalCrossentropy': lambda **kwargs: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum_over_batch_size')}
    model = models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("[INFO] Loaded model from disk.")

    image = cv2.imread(image_path, 0)      
    image1 = cv2.resize(image, (28,28))    
    _, thresh = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    image2 = image1.reshape(1,28,28,1)

    pred = np.argmax(model.predict(image2), axis=-1)
    return pred[0]   

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        

    def create_widgets(self):
        self.load_button = Button(self, text="Load Image", command=self.load_image)
        self.load_button.pack(side="top")

        self.image_label = Label(self)
        self.image_label.pack(side="top")

        self.predict_button = Button(self, text="Predict Digit", command=self.predict_digit)
        self.predict_button.pack(side="top")

        self.result_label = Label(self, text="")
        self.result_label.pack(side="top")

        self.quit = Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")

        self.image_path = ""

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            print(f"Loaded image: {self.image_path}")
            # Display the image in the GUI
            img = PIL.Image.open(self.image_path)
            img = img.resize((200, 200))
            photo = PIL.ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        else:
            print("No image loaded")

    def predict_digit(self):
        if self.image_path:
            predicted_digit = predict_digit(self.image_path)
            self.result_label.config(text=f"Predicted Digit: {predicted_digit}")
        else:
            self.result_label.config(text="No image loaded")

root = tk.Tk()
app = Application(master=root)
app.mainloop()