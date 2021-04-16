import tkinter as tk
import numpy as np
from tkinter import filedialog, Label, Button, BOTTOM
import cv2
from PIL import ImageTk, Image
import numpy
import os
import skimage.transform as st
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import load_model
model = load_model('data5.pickle_epoki_50_ilosc_danych_86000_partia_128_sheduler_dflt.h5')
classes = {1: 'Speed limit (20km/h)', 
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

def camera_function(model):
    
    frameWidth= 480
    frameHeight = 640
    brightness = 180
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)

    while True:
        xxx, imgOrignal = cap.read()
        x_g = np.zeros((480, 640, 1))
        x_g[:, :, 0] = imgOrignal[:, :, 2] * 0.299 + imgOrignal[:, :, 1] * 0.587 + imgOrignal[:, :, 0] * 0.114
        x_g = list(map(hist_eq, x_g[:, :, 0].astype(np.uint8)))
        img2 = np.asarray(x_g)
        img = st.resize(img2, (32, 32), preserve_range=True)
        img = np.asarray(img).astype('uint8')
        img = np.expand_dims(img, axis=0)
        cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "press 'q' to close" , (20, 460), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        classIndex = model.predict_classes(img)
        val = np.amax(model.predict(img), axis=1)[0]
        if val > 0.95:
            cv2.putText(imgOrignal,str(classIndex + 1) + " " + str(classes[classIndex[0]+1]), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(val* 100, 2) ) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Result', imgOrignal)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def live_feed():
    camera_function(model)


def hist_eq(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(image)

def classify(file_path):
    image = Image.open(file_path)
    filepathsave = file_path
    print(filepathsave)
    image = image.resize((32, 32))
    image = numpy.array(image)
    x_g = np.zeros((1, 32, 32))
    x_g[0, :, :] = image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114
    x_g = list(map(hist_eq, x_g[0, :, :].astype(np.uint8)))
    x_g = numpy.expand_dims(x_g, axis=0)
    x_g = np.array(x_g)
    x_g = x_g.reshape(1, 1, 32, 32)
    x_g = x_g.transpose(0, 2, 3, 1)
    pred = np.argmax(model.predict([x_g]), axis=-1)[0]
    sign = classes[pred + 1]
    print(sign)
    label.configure(foreground='black', text=sign, font=('arial', 20, 'bold'))


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Sign", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#6699ff', foreground='black', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        img = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=img)
        sign_image.image = img
        show_classify_button(file_path)
    except:
        pass


top = tk.Tk()
top.geometry('800x800')
top.title('Traffic sign recognition')
top.configure(background='#d9d9d9')
label = Label(top, foreground='black', background='#d9d9d9', font=('arial', 20, 'bold'))
sign_image = Label(top)
upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#6699ff', foreground='black', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)

live_vid_feed = Button(top, text="Video Feed", command=live_feed, padx=10, pady=5)
live_vid_feed.configure(background='#6699ff', foreground='black', font=('arial', 10, 'bold'))
live_vid_feed.pack(side=BOTTOM)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Traffic sign recognition", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#d9d9d9', foreground='black')
heading.pack()

top.mainloop()