import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import requests


root = tk.Tk()
root.title("Mole Classification")
root.geometry("400x400")


image_label = tk.Label(root)
image_label.pack()


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((300, 300))  
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk  
            classify_image(file_path)
        except Exception as e:
            result_label.config(text=f"Error opening image: {e}")


def classify_image(file_path):
    url = "http://127.0.0.1:5000/predict"
    files = {'file': open(file_path, 'rb')}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        data = response.json()
        prediction = data.get('prediction', 'No prediction')
        confidence = data.get('confidence', 0)
        result_label.config(text=f"Prediction: {prediction}\nConfidence: {confidence:.2f}")
    except requests.exceptions.RequestException as e:
        result_label.config(text=f"Error: {e}")


btn = tk.Button(root, text="Open Image", command=open_image)
btn.pack()


result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
