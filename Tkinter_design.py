import Decision_Tree as dt
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

window = Tk()
window.geometry("1000x600")
window.title('Decision Tree')

#fn = StringVar()
# Catch event choose of OptionMenu
var = StringVar()
filename = 'null'
play = []
columns = []
label_encoder = LabelEncoder()
dTreeClf = DecisionTreeClassifier()
Om_arr = []
var_arr = []
predict_arr = []
count = 0


def prints():
    global count, filename, play, columns, label_encoder, dTreeClf, var, Om_arr, var_arr
    filename = filename.replace('/', '\\')
    count, play, columns, label_encoder, dTreeClf = dt.train_model(filename)
    load = Image.open("D:\\Work_Space\\Machine_Learning\\Decision_Tree\\diabetes.png")
    load = load.resize((500, 500), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img = Label(window, image=render)
    img.image = render
    img.place(x=0, y=30)
    label5 = Label(window, text='Accuracy is : ' + str(count), fg='blue',
                   bg='yellow', font=('arial', 16, 'bold')).place(x=600, y=100)
    y_length = 300

    # Count number of Option Menu
    count = 0
    # Array of Option MenuOm_arr = []
    for i in play.columns:
        list1 = set(play[i])
        # print(list1)
        var_arr.append(var)
        droplist = OptionMenu(window, var_arr[count], *list1)
        Om_arr.append(droplist)
        var_arr[count].set(str(i))
        Om_arr[count].config(width=20)
        Om_arr[count].place(x=600, y=y_length)
        y_length += 40
        count += 1
        var = StringVar()
    #cookImage.place(x=100, y=200)
    b3 = Button(window, text='Predict', width=20, bg='red', fg='white', command=predict)
    y_length += 40
    b3.place(x=600, y=y_length)


def predict():
    global predict_arr, play, columns, label_encoder, dTreeClf
    # print(play)
    predict_arr = []
    for i in var_arr:
        if i.get() in play.columns:
            predict_arr.append(np.NaN)
            continue
        predict_arr.append(i.get())
    test_play = pd.DataFrame([predict_arr])
    test_play.columns = play.columns
    test_play = pd.get_dummies(test_play, columns=test_play.columns)
    missing_column = set(columns) - set(test_play.columns)
    for c in missing_column:
        test_play[c] = 0
    X_test = test_play[columns]
    predictions = dTreeClf.predict(X_test)
    predictions = label_encoder.inverse_transform(predictions)
    result = 'The result is ' + predictions[0]
    label = Label(window, text=result, width=14, fg='blue',
                  bg='yellow', font=('arial', 16, 'bold')).place(x=800, y=500)


def prints_get(event):
    print(var.get())


def upload_File():
    global filename
    filename = filedialog.askopenfilename()
    # print(type(filename))


    # Label
label = Label(window, text='Decision Tree', fg='blue',
              bg='yellow', font=('arial', 16, 'bold')).pack()
# Entry
#entry = Entry(window, textvar=fn).place(x=240, y=242)

b1 = Button(window, text='Upload Data Train', width=20, bg='red', fg='white', command=upload_File)
b1.place(x=600, y=200)

b1 = Button(window, text='Train', width=10, bg='red', fg='white', command=prints)
b1.place(x=800, y=200)
# Drop List
window.mainloop()
