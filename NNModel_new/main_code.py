import numpy as np
import random
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time

model = load_model("RNN_model_resolved100")

input = []
output = []
i = 0
n_past = 14
input_init = []
forecasting = []
while True:
    if i < n_past:
        ran = random.randint(600,3000)
        input_init.append(ran)
        input_zero = np.zeros((n_past-(i+1),))
        input_zero = input_zero.tolist()

        input_totall = input_init + input_zero
        print('VALUES',input_totall,'type',np.shape(input_totall))
        input_total = np.array(input_totall)
        input_total = np.reshape(input_total,(1,n_past,1))

        forecastt = model.predict(input_total) #shape = (n, 1) where n is the n_days_for_forecast
        forecastt = forecastt.tolist()[0][0]
        forecasting.append(forecastt)

        fig, ax_left = plt.subplots()
        ax_left.plot(list(range(len(forecasting))),forecasting,'-go', label = 'well pred')
        ax_left.set_ylabel('well pred')

        ax_right = ax_left.twinx()
        ax_right.plot(list(range(len(input_total[0,:,:]))),input_total[0,:,:], label = 'glir')
        ax_right.set_ylabel('glir')
        ax_left.legend()
        ax_right.legend()
        ax_left.grid()
        
        plt.pause(0.05)
        fig.clear()

        input_total = input_init + input_zero
        i+=1
    else:
        ran = random.randint(600,3000)
        input_totall.append(ran)
        print('VALUES',input_totall,'type',np.shape(input_totall))
        input_total = input_totall[-n_past:]
        input_total = np.array(input_total)
        input_total = np.reshape(input_total,(1,n_past,1))
        print('VALUE',input_total,'type',np.shape(input_total))

        forecastt = model.predict(input_total) #shape = (n, 1) where n is the n_days_for_forecast
        forecastt = abs(forecastt)
        forecastt = forecastt.tolist()[0][0]
        forecasting.append(forecastt)

        #print("done")
        #print('forecasted:',forecasting)
        fig, ax_left = plt.subplots()
        ax_left.plot(list(range(len(forecasting))),forecasting,'-go', label = 'well pred')
        ax_left.set_ylabel('well pred')

        ax_right = ax_left.twinx()
        ax_right.plot(list(range(len(input_totall[:]))),input_totall[:], label = 'glir')
        ax_right.set_ylabel('glir') 
        ax_left.legend()
        ax_right.legend()
        ax_left.grid()
        
        plt.pause(0.05)
        fig.clear()
        i+=1
        #break
    #plt.show()