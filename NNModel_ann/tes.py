from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model

"""file_pre = 'C:/Users/ASUS/Documents/AllThingsPython/GLO_NN/NNModel_ann/RNN_wc1_type4_20+300'
file_pretrained = file_pre.replace('/','\\')
print(file_pretrained)
model = load_model(f"{file_pretrained}")
print(str(model.summary("")))

#plt.rc('figure', figsize=(8, 5))
plt.text(0.01, 0.05, str(model.summary("")), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
#plt.savefig('results.png')
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)"""

import pandas as pd

df1 = pd.read_excel("oa-11.xlsx")
df2 = pd.read_excel("oa-12.xlsx")

print(df1.describe())
print(df2.describe())