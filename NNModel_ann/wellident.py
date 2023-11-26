import tkinter as tk
from tkinter import ttk #widget
from tkinter import filedialog
from threading import Thread
from time import sleep
from PIL import ImageTk, Image

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import r2_score

from tensorflow.keras.models import load_model
from keras.utils.vis_utils import plot_model
import seaborn as sns

# ======================================== INITIAL ========================================
window = tk.Tk()
window.configure(bg='white')
window.geometry("1700x1050")
#window.resizable(False,False)
window.title("Neural Network Well Dynamics")

# ======================================== FRAME ========================================
frame_header = tk.Frame(window, bg='white')
frame_header.grid(row=0,column=0)

frame_content = tk.Frame(window, bg='white')
frame_content.grid(row=1,column=0)

frame_comp = tk.Frame(window, bg='white')
frame_comp.grid(row=1,column=1)

frame_input = tk.Frame(frame_content,bg='white')
frame_input.grid(row=0,column=0)

frame_display = tk.Frame(frame_content, bg='white')
frame_display.grid(row=1,column=0,pady=2)

# ======================================== HEADER LABEL ========================================
img = ImageTk.PhotoImage(Image.open("header.png"))
dragdown_label = tk.Label(frame_header,image = img)
dragdown_label.pack()

# ======================================== HEADER FRAME ========================================
input_label = tk.LabelFrame(frame_input, text="PARAMETERS",bg='white')
input_label.grid(row=0,column=0,padx=10,pady=10)

# ================= HEADER FILLING =================
input_label_file = tk.LabelFrame(input_label, text="WELL TEST DATA")
input_label_file.grid(row=0,column=0,padx=10,pady=10)
filename = 'upsampled_corr_well.csv'

def openFile():
    global filename
    filepath = filedialog.askopenfilename(title="Select Well Testting Production Data",
                                            filetypes=[("CSV files",'.csv')])
    filename = filepath.split('/')[-1]
    filelog.config(text=filename,font=('bold'))
    print(filepath)
    print(filename)
    
    df = pd.read_csv(filename)
    print(df)

set_file = tk.Button(input_label_file,text='Select File',command=openFile)
set_file.grid(row=0,column=0,padx=10,pady=10)
filelog = tk.Label(input_label_file,text=filename,bg='yellow')
filelog.grid(row=0,column=1,padx=10,pady=10)

def select(event):
    print(f"value:{click.get()}")
option_nn = ["All",
            "Gas Lift Injection Rate",
            "Water Cut",
            "Casing Head Pressure",
            "Gas Oil Ratio",
            "GLIR - WC",
            "GLIR - WC - CHP"]
click = tk.StringVar()
click.set(option_nn[0])

feature_label = tk.Label(input_label_file, text="Reservoir Feature *")
feature_label.grid(row=1,column=0,padx=0,pady=0)

dropdown_nn = tk.OptionMenu(input_label_file, click, *option_nn, command=select)
dropdown_nn.grid(row=2,column=0,padx=0,pady=0)


def selected(event):
    print(f"value:{clicked.get()}")
welltype = ''
options = ["Oil Flow Rate",
            "Liquid Flow Rate",
            "Gas Lift Injection Rate",
            "Water Cut",
            "Casing Head Pressure",
            "Gas Oil Ratio"]
clicked = tk.StringVar()
clicked.set(options[0])

target_label = tk.Label(input_label_file, text="Reservoir Target *")
target_label.grid(row=1,column=1,padx=0,pady=0)

dropdown = tk.OptionMenu(input_label_file, clicked, *options, command=selected)
dropdown.grid(row=2,column=1,padx=0,pady=0)

def pretrained():
    global file_pretrained
    file_pre = filedialog.askdirectory(title="Select RNN MODEL")
    file_pretrained = file_pre.replace('/','\\')

    print(file_pretrained)

button_pretrained = tk.Button(input_label, text='Select Pre-trained Model',command=pretrained)
button_pretrained.grid(row=6,column=0,padx=10,pady=10)

input_label_split = tk.LabelFrame(input_label, text="SPLIT RATIO (%)")
input_label_split.grid(row=1,column=0,padx=5,pady=5)
set_split = tk.Entry(input_label_split)
set_split.insert(0, "70")
set_split.grid(row=0,column=0,padx=5,pady=5)

input_label_epoch = tk.LabelFrame(input_label, text="EPOCH")
input_label_epoch.grid(row=2,column=0,padx=5,pady=5)
set_epoch = tk.Entry(input_label_epoch)
set_epoch.insert(0, "500")
set_epoch.grid(row=0,column=0,padx=5,pady=5)

input_label_batchsize = tk.LabelFrame(input_label, text="BATCH SIZE")
input_label_batchsize.grid(row=3,column=0,padx=5,pady=5)
set_batchsize = tk.Entry(input_label_batchsize)
set_batchsize.insert(0, "300")
set_batchsize.grid(row=0,column=0,padx=5,pady=5)

input_label_save = tk.LabelFrame(input_label, text="Model Name")
input_label_save.grid(row=4,column=0,padx=5,pady=5)
set_save = tk.Entry(input_label_save)
set_save.insert(0, f"RNN_{clicked.get()}_{int(set_epoch.get())}+{int(set_batchsize.get())}")
set_save.grid(row=0,column=0,padx=5,pady=5)

def fit():
    global imgg
    csv_file = filename
    split = int(set_split.get())
    split = split/100
    epoch = int(set_epoch.get())
    batchsize = int(set_batchsize.get())
    param = clicked.get()
    params = click.get()
    save_file = set_save.get()

    df = pd.read_csv(csv_file)
    data = df['glir'].to_numpy()

    if params == "Gas Lift Injection Rate":
        features = ['glir']
    elif params == "Water Cut":
        features = ['wc']
    elif params == "Casing Head Pressure":
        features = ['ch']
    elif params == "Gas Oil Ratio":
        features = ['gor']
    elif params == "GLIR - WC":
        features = ['glir','wc']
    elif params == "GLIR - WC - CHP":
        features = ['glir','wc','ch']
    elif params == "All":
        features = ['glir','wc','ch','gor']
    
    if param == "Oil Flow Rate":
        target = ['qo']
        axiss = "Oil Flow Rate (STB/Day)"
    elif param =="Liquid Flow Rate":
        target = ['qt']
        axiss = "Liquid Flow Rate (STB/Day)"
    elif param == "Gas Lift Injection Rate":
        target = ['glir']
        axiss = "Gas Lift Injection Rate (MSCFD)"
    elif param == "Water Cut":
        target = ['wc']
        axiss = "Water Cut (%)"
    elif param == "Casing Head Pressure":
        target = ['ch']
        axiss = "Casing Head Pressure (psia)"
    elif param == "Gas Oil Ratio":
        target = ['gor']
        axiss = "Gas Oil Ratio (SCF/STB)"
        

    x = df[features]
    y = df[target]

    x1 = df[features][:int(split*len(data))]
    y1 = df[target][:int(split*len(data))]

    x2 = df[features][int(split*len(data)):]
    y2 = df[target][int(split*len(data)):]


    print('feat',features,len(features))
    print('tar',len(target))

    # ================= TRAINING =================

    scaler = StandardScaler()

    df_for_x = scaler.fit_transform(x)
    df_for_y = scaler.fit_transform(y)

    X = df_for_x
    Y = df_for_y

    trainX, testX, trainY, testY = train_test_split(X,Y,test_size=1-split,random_state=123,shuffle=False)
    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))
    print('testX shape == {}.'.format(testX.shape))
    print('testY shape == {}.'.format(testY.shape))

    lookback = 14
    win_len = batchsize
    num_feature = np.shape(x)[1]

    train_generator = TimeseriesGenerator(trainX,trainY,batch_size=win_len,length=lookback)
    test_generator = TimeseriesGenerator(testX,testY,batch_size=win_len,length=lookback)

    # ================= MODEL RNN =================
    model = Sequential()
    model.add(LSTM(14,activation='relu',input_shape=(lookback, num_feature), return_sequences=True))
    #model.add(LSTM(7,activation='relu', return_sequences=True))
    #model.add(LSTM(4,activation='relu', return_sequences=False))
    model.add(LSTM(50,activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    #print(model.summary(""))

    if save_file != '':
        print("=========TRAINING PROCESS=========")
        history = model.fit(train_generator, 
                            validation_data = test_generator, 
                            epochs=epoch, 
                            batch_size=batchsize, 
                            verbose=1,
                            shuffle=False
                            )
        model.save(save_file)
    else:
        print("=========PRE-TRAINED MODEL=========")
        model = load_model(f"{file_pretrained}")
        print(f"LOADED MODEL: {model}")
        #if len(features) == 1:
        #    model = load_model(f"{file_pretrained}")
        #else:
        #    model = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\RNN_qo_type2_20+300') #OA-11
            #model = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\RNN_qo_type4_20+300') #OA-12
            #history = model.fit(train_generator, validation_data = test_generator, epochs=epoch, batch_size=200, verbose=1,shuffle=False)
    
    print(model.summary(""))
    
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    imgg = ImageTk.PhotoImage(Image.open("model_plot.png"))
    textarea = tk.Label(structure_frame, image=imgg)
    textarea.grid(row=0,column=0)
    #textarea.config(text=str(model.summary("")))

    # ================= TRAIN RESULT =================
    #y_pred_train = model.predict_generator(train_generator)
    y_pred_train = model.predict(train_generator)
    #y_pred_train = np.reshape(y_pred_train,(np.shape(y_pred_train)[0],14))
    df_pred_train = pd.concat([pd.DataFrame(y_pred_train), pd.DataFrame(trainX[:,:][lookback:])],axis=1)

    #rev_trans_train = scaler.inverse_transform(df_pred_train)
    rev_trans_train = scaler.inverse_transform(y_pred_train)

    y_predict_train = np.resize(rev_trans_train[:,0],(len(rev_trans_train[:,0]),1))
    y_real = y1[lookback:]

    df_final_train = pd.concat([pd.DataFrame(y_predict_train), pd.DataFrame(y_real)],axis=1)
    df_final_train
    print(y_predict_train.shape)
    print(y_real.shape)

    if save_file != '':
        ukuran = 130
        Figure().clf()
        plot_fig = Figure(figsize=(8,4))
        
        ax2 = plot_fig.add_subplot(ukuran+3)
        xx = np.arange(0,len(history.history['loss']))
        ax2.plot(xx,history.history['loss'], label='Training loss')
        ax2.plot(xx,history.history['val_loss'], label='Validation loss')
        ax2.set_title("Training Loss vs Validation Loss")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("val")
        ax2.legend()
        ax2.grid()
    else:
        ukuran = 120
        Figure().clf()
        plot_fig = Figure(figsize=(8,4))

    #Figure().clf()
    #plot_fig = Figure(figsize=(8,4))
    ax = plot_fig.add_subplot(ukuran+1)
    ax.set_title("RNN Prediction: Train")
    ax.plot(np.arange(0,len(y_predict_train)),y_predict_train,'-*',label='pred')
    ax.plot(np.arange(0,len(y_real)),y_real,'-o',label='real')
    ax.plot()
    ax.set_xlabel('Days')
    #ax.set_ylabel("Oil Flow Rate (STB/day)")
    ax.set_ylabel(axiss)
    ax.grid(True)
    ax.legend()

    r2 = r2_score(y_real,y_predict_train)
    print("\nR2 Value:")
    print(round(r2,2))
    MSE = np.square(np.subtract(y_real,y_predict_train)).mean()  
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error:")
    print(round(RMSE,2))

    ax1 = plot_fig.add_subplot(ukuran+2)
    ax1.scatter(y_real,y_predict_train)
    ax1.plot(y_real,y_real,'g',label = '1D Fitting Validation')
    mymodel = np.poly1d(np.polyfit(y_predict_train.flatten(),np.array(y_real).flatten(), 1))
    myline = np.linspace(min(np.array(y_real)), max(np.array(y_real)))
    ax1.plot(myline, mymodel(myline), color="orange", label=f'1D Fitting Prediction of {mymodel}')
    #plt.xlabel("Validation Data")
    #plt.ylabel("Prediction Data")
    #ax1.set_title(f"(R2 Score, RMSE) = ({round(r2,2)},{round(RMSE,2)}) of Training data")
    #ax1.set_title(f"RMSE = ({round(RMSE,2)})")
    ax1.set_title(f"RMSE ; R2 = ({round(RMSE,2)}) ; ({round(r2,2)})")
    ax1.set_xlabel('Real Data')
    ax1.set_ylabel('Prediction Data')
    ax1.grid(True)
    ax1.legend()

    canvas = FigureCanvasTkAgg(plot_fig,master=training_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=0,padx=0,pady=0) 


    # ================= VALIDATION RESULT =================
    #y_pred_test = model.predict_generator(test_generator)
    y_pred_test = model.predict(test_generator)
    #y_pred_test = np.reshape(y_pred_test,(np.shape(y_pred_test)[0],14))
    rev_trans_test = scaler.inverse_transform(y_pred_test)

    y_predict_test = np.resize(rev_trans_test[:,0],(len(rev_trans_test[:,0]),1))
    y_real_test = y2[lookback:]

    df_final_test = pd.concat([pd.DataFrame(y_predict_test), pd.DataFrame(y_real_test)],axis=1)
    df_final_test
    print(y_predict_test.shape)
    print(y_real_test.shape)

    Figure().clf()
    plot_fig2 = Figure(figsize=(8,4))
    ax = plot_fig2.add_subplot(121)
    ax.set_title("RNN Prediction: Validation")
    ax.plot(np.arange(0,len(y_predict_test)),y_predict_test,'-*',label='pred')
    ax.plot(np.arange(0,len(y_real_test)),y_real_test,'-o',label='real')
    #ax.set_ylabel("Oil Flow Rate (STB/day)")
    ax.set_ylabel(axiss)
    ax.grid(True)
    ax.legend()

    r2 = r2_score(y_real_test,y_predict_test)
    print("\nR2 Value:")
    print(round(r2,2))
    MSE = np.square(np.subtract(y_real_test,y_predict_test)).mean()  
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error:")
    print(round(RMSE,2))

    ax1 = plot_fig2.add_subplot(122)
    ax1.scatter(y_real_test,y_predict_test)
    ax1.plot(y_real_test,y_real_test,'g',label = '1D Fitting Validation')

    mymodel = np.poly1d(np.polyfit(y_predict_test.flatten(),np.array(y_real_test).flatten(), 1))
    myline = np.linspace(min(np.array(y_real_test)), max(np.array(y_real_test)))
    ax1.plot(myline, mymodel(myline), color="orange", label=f'1D Fitting Prediction of {mymodel}')
    #plt.xlabel("Validation Data")
    #plt.ylabel("Prediction Data")
    #ax1.set_title(f"(R2 Score,RMSE) = ({round(r2,2)},{round(RMSE,2)}) of Validation data")
    ax1.set_title(f"RMSE ; R2 = ({round(RMSE,2)}) ; ({round(r2,2)})")
    ax1.set_xlabel('Real Data')
    ax1.set_ylabel('Prediction Data')
    ax1.grid(True)
    ax1.legend()
    
    canvas = FigureCanvasTkAgg(plot_fig2,master=validation_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=0,padx=0,pady=0)
    
    print(split,epoch,batchsize,csv_file,params,x,y)


#text_frame = tk.LabelFrame(frame_display, text="DISPLAY",bg='white')
#text_frame.grid(row=0,column=0)
structure_frame = tk.LabelFrame(frame_input, text="RNN STRUCTURE",bg='white')
structure_frame.grid(row=0,column=2,padx=10,pady=10)

#imgg = ImageTk.PhotoImage(Image.open("model_plot.png"))
#textarea = tk.Label(structure_frame, image=imgg)
#textarea.grid(row=0,column=0)

result_frame = tk.LabelFrame(frame_input, text="MODEL RESULT",bg='white')
result_frame.grid(row=0,column=1,padx=10,pady=10)

training_frame = tk.LabelFrame(result_frame, text="TRAINING RESULT",bg='white')
training_frame.grid(row=0,column=0,padx=5,pady=5)
#train_pack = tk.LabelFrame(training_frame)
#train_pack.grid(row=0,column=1,padx=0,pady=0)

validation_frame = tk.LabelFrame(result_frame, text="VALIDATION RESULT",bg='white')
validation_frame.grid(row=1,column=0,padx=5,pady=5)
#val_pack = tk.LabelFrame(validation_frame)
#val_pack.grid(row=0,column=1,padx=0,pady=0)


button_confirm = tk.Button(input_label, text='FIT',command=fit)
button_confirm.grid(row=5,column=0,padx=10,pady=10)

# ======================================== MAIN PROGRAM ========================================

window.mainloop()	