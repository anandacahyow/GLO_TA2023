import argparse
import csv
from datetime import datetime
import logging
from math import nan
import threading
import pandas
import random
import math
import numpy as np


from time import sleep
from pymodbus.server.sync import StartSerialServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSparseDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.transaction import ModbusRtuFramer
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder


from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler


logging.basicConfig(level=logging.INFO)


# ========================================== MODEL INITIALIZATION  ==========================================
#4 PARAMS BASED
model = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\3LSTM_ALL_QO1_15012090')
model1 = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\3LSTM_ALL_QO2_15012090')


print(f"\n well OA-11 Model: <{model}> has been loaded\n")
print(f"\n well OA-12 Model: <{model1}> has been loaded\n")


# ========================================== Model Params ==========================================
model_wc1 = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\model_SIMPLE_WC1_90')
model_ch1 = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\model_SIMPLE_CHP1_90')
model_gor1 = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\model_SIMPLE_GOR1_90')


model_wc2 = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\model_SIMPLE_WC2_90')
model_ch2 = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\model_SIMPLE_CHP2_90')
model_gor2 = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_ann\model_SIMPLE_GOR2_90')


print("===== done importing models =====")

df = pandas.read_csv("upsampled.csv")
x = df[['glir11','wc11','ch11','gor11']]
y = df[['qo11']]


df2 = pandas.read_csv("upsampled_corr.csv")
x2= df2[['glir22','wc22','ch22','gor22']]
y2 = df2[['qo22']]


o= []
p= []
q= []
r= []
h= []
i= []
j= []
k= []
for whole in range(0,14):
    o.append(random.randint(54,676))
    p.append(random.randint(4,63))
    q.append(random.randint(465,1767))
    r.append(random.randint(7615,21955))


    h.append(random.randint(10,6220))
    i.append(random.randint(1,60))
    j.append(random.randint(392,1694))
    k.append(random.randint(8,107361))

print('glir=',o)


# ========================================== MODBUS SETTING ==========================================


class ModbusEntry:
    __r_length = {
        "UNSIGNED INT": 1,
        "INT": 1,
        "FLOAT": 2,
        "DISCRETE": 1,
        "BOOLEAN": 1,
        "BYTE": 1,
        "UNSIGNED BYTE": 1,
        "DOUBLE": 4,
        "LONG": 2,
        "SIGNED LONG": 2,
        "UNSIGNED LONG": 2,
    }


    __endiannes = {
        "4321": {
            "byte_order": Endian.Big,
            "word_order": Endian.Big,
        },
        "1234": {
            "byte_order": Endian.Little,
            "word_order": Endian.Little,
        },
        "2143": {
            "byte_order": Endian.Big,
            "word_order": Endian.Little,
        },
        "3412": {
            "byte_order": Endian.Little,
            "word_order": Endian.Big,
        },
    }


    __DESCRIPTION = 0  # string
    __IO_NAME = 1  # string
    __DATA_TYPE = 2  # UNSIGNED INT, FLOAT, DISCRETE, BOOLEAN, INT, SIGNED INT, BYTE, UNSIGNED BYTE, DOUBLE, LONG, SIGNED LONG, UNSIGNED LONG
    __REGISTER_TYPE = 3  # 0, 10000, 30000, 40000 | discrete, coil, input, holding
    __REGISTER_OFFSET = 4  # 4 number
    __ENDIANNES = 5  # 4321, 1234, 3412, 2143
    __IO_TYPE = 6  # R, W, RW
    __UNIT = 8


    def __init__(self, row):
        try:
            if len(row) < 6:
                raise Exception("row data length less than required")
            self.description = str(row[self.__DESCRIPTION]).strip()
            self.io_name = str(row[self.__IO_NAME]).strip()
            self.data_type = str(row[self.__DATA_TYPE]).strip()
            self.register_length = self.__r_length[str(
                row[self.__DATA_TYPE]).strip()]
            self.register_type = str(row[self.__REGISTER_TYPE]).strip()
            self.register_offset = str(row[self.__REGISTER_OFFSET]).strip()
            self.endiannes = self.__endiannes[str(int(row[self.__ENDIANNES]))]
            self.io_type = str(row[self.__IO_TYPE])
            self.unit = str(row[self.__UNIT])
        except:
            pass


class ModbusTemplate:
    def __init__(self, path):
        """try:
            excel_data = pandas.read_excel(path, skiprows=3)
        except:
            print('FAILED READING')"""
        excel_data = pandas.read_excel(path, skiprows=3)
        self.modbus_entries = []
        for row in excel_data.values:
            entry = ModbusEntry(row)
            self.modbus_entries.append(entry)


class SlaveContexts:
    def __init__(self, slave_ids,modbus_templates,val_data):
        self.contexts = {}  # dictionary of ModbusSlaveContext, key = slave_id
        #self.val_data = {}
        try:
            self.__templates = (modbus_templates)  # dictionary of ModbusTemplate, key = slave_id
            self.__val_data = (val_data)


            for id in slave_ids:
                ctx = ModbusSlaveContext(
                    di=ModbusSparseDataBlock(),
                    co=ModbusSparseDataBlock(),
                    hr=ModbusSparseDataBlock(),
                    ir=ModbusSparseDataBlock(),
                )
                self.contexts[int(id)] = ctx


        except:
            logging.error("failed in instantiating SparseDataBlock class")
   
    def update_context(self, slave_id,val_data):
        ctx = self.contexts[slave_id]
        template = self.__templates[slave_id]
        values = self.__val_data[slave_id]
        #print(f"NILAI SLAVE inside loop: {values}")
       
        for entry in template.modbus_entries:
            if slave_id == 113:
                if entry.register_type == "30000" and int(entry.register_offset) == 167: # Qw VX
                    simulated_value = round(values[0],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 191: # Wc
                    simulated_value = round(values[1],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 111: # Qw_lc
                    simulated_value = round(values[2],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 165: # Qo VX
                    simulated_value = round(values[3],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 109: # Qo_lc
                    simulated_value = round(values[4],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 105: # PL (pressure line)
                    simulated_value = round(values[5],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 199: # GOR
                    simulated_value = round(values[6],3)
                else:
                    continue
            elif slave_id ==114:
                if entry.register_type == "40000" and int(entry.register_offset) == 7031: #GLIR ABB
                    simulated_value = int(values)
                else:
                    continue
            elif slave_id == 115:
                if entry.register_type == "30000" and int(entry.register_offset) == 167: # Qw VX
                    simulated_value = round(values[0],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 191: # Wc
                    simulated_value = round(values[1],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 111: # Qw_lc
                    simulated_value = round(values[2],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 165: # Qo VX
                    simulated_value = round(values[3],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 109: # Qo_lc
                    simulated_value = round(values[4],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 105: # PL (pressure line)
                    simulated_value = round(values[5],3)
                elif entry.register_type == "30000" and int(entry.register_offset) == 199: # GOR
                    simulated_value = round(values[6],3)
                else:
                    continue
            elif slave_id ==116:
                if entry.register_type == "40000" and int(entry.register_offset) == 7031: #GLIR ABB
                    simulated_value = int(values)
                else:
                    continue
            else:
                continue


                       
            register_values = self.__pack_by_endiannes(simulated_value, entry.endiannes, entry.data_type)




            if entry.register_type == "0":
                ctx.store["d"].setValues(int(entry.register_offset)+1, register_values)
            elif entry.register_type == "10000":
                ctx.store["c"].setValues(int(entry.register_offset)+1, register_values)
            elif entry.register_type == "30000":
                ctx.store["i"].setValues(int(entry.register_offset)+1, register_values)
            elif entry.register_type == "40000":
                ctx.store["h"].setValues(int(entry.register_offset)+1, register_values)
            # ========================================== LOG VALUEs ==========================================
            #print(f"SLVAE-{slave_id} REG VAL:{register_values} REGIST: {int(entry.register_type) + int(entry.register_offset)}")


            logging.info(
                    "{} slave: {} | tag: {} \t- {}\t | value: {} {}".format(
                        datetime.now(),
                        slave_id,
                        entry.io_name,
                        int(entry.register_type) + int(entry.register_offset),
                        simulated_value,
                        entry.unit
                    )
                )


    def __pack_by_endiannes(self, val, endiannes, data_type):
        builder = BinaryPayloadBuilder(
            byteorder=endiannes["byte_order"], wordorder=endiannes["word_order"]
        )


        if val == "":  # set to 0 if empty string
            val = 0


        if data_type == "UNSIGNED INT":
            builder.add_16bit_uint(int(val))
        elif data_type == "INT":
            builder.add_16bit_int(int(val))
        elif data_type == "FLOAT":
            builder.add_32bit_float(float(val))
        elif data_type == "DISCRETE":
            builder.add_bits(int(val))
        elif data_type == "BOOLEAN":
            builder.add_bits(int(val))
        elif data_type == "BYTE":
            builder.add_8bit_int(int(val))
        elif data_type == "UNSIGNED BYTE":
            builder.add_8bit_uint(int(val))
        elif data_type == "DOUBLE":
            builder.add_64bit_float(float(val))
        elif data_type == "LONG":
            builder.add_32bit_int(int(val))
        elif data_type == "UNSIGNED LONG":
            builder.add_32bit_uint(int(val))


        return builder.to_registers()


def WellDyn(GLIR,a,b,c,e):
    Qt = a*(GLIR**2) + b*(GLIR) + c
    Qt_rand = Qt + Qt*random.uniform(-e/100, e/100)
    return Qt_rand


def WC_dyn(t):
    wc = ((0.15)/(1+20*(math.exp(-0.001*t))))+0.6
    wc2 = ((0.2)/(1+20*(math.exp(-0.001*t))))+0.57
    #wc = 0.75
    return [wc,wc2]


def WellSys(u,i,mode):
    import control as ctl
   
    if np.shape(u) == (1,):
        x_ident = [0,0,u[0]]
    elif np.shape(u) == (2,):
        x_ident = [0,u[0],u[1]]
    else:
        x_ident1 = [0,u[0],u[1]]
        x_ident_temp = x_ident1.copy()
        x_ident = x_ident_temp + u[2:i+1]


    u_sys = x_ident
   
    # ========================================== WELL DYNAMICS VARIATIONS ==========================================
    #Third Order Transfer Function Delay of 2
    num = np.array([0,0,0.190904336050159,-0.189899401826588])
    den = np.array([1,-1.221349138175800,0.100981241074881,0.123753106411842])


    K = 1
    Ts = 1  #1 day sampling day
    sys = ctl.TransferFunction(K*num,den, dt=Ts)


    res = ctl.forced_response(sys,T=None,U=u_sys,X0=0)
    y_sys = res.outputs
    x_sys = res.inputs


    if mode == 'qt':
        y_sys  = y_sys + y_sys*random.uniform(-5/100, 5/100)
    elif mode == 'qo':
        y_sys = y_sys*(1 - WC_dyn(i)[0])
    #print(sys)
    return y_sys


def NN_Model(model, well_name, glir, wc, cashead, gor, lookback, ref):
    scaler = StandardScaler()
    scale = StandardScaler().fit(ref)
   
    """if well_name == 'OA-11':
        glir.append(random.randint(54,676))
        wc.append(random.randint(4,63))
        cashead.append(random.randint(465,1767))
        gor.append(random.randint(7615,21955))
    elif well_name == 'OA-22':
        glir.append(random.randint(10,6220))
        wc.append(random.randint(0.14,60))
        cashead.append(random.randint(392,1694))
        gor.append(random.randint(8,107361))"""
       
    glir = glir[-14:]
    wc = wc[-14:]
    cashead = cashead[-14:]
    gor = gor[-14:]


    tot = np.stack((glir,wc,cashead,gor),axis=1).reshape(1,-1,4)
    Tot = scaler.fit_transform(tot[0])
    Tot = Tot.reshape(1,14,4)
    #print(tot)


    s = model.predict(Tot) #shape = (n, 1) where n is the n_days_for_forecast
    forecastt = scale.inverse_transform(s)[0][0]


    value_simulated = forecastt
    return value_simulated


def NN_Model_Params(model,feature,lookback,ref):
    scaler = StandardScaler()
    scale = StandardScaler().fit(ref)


    feature = feature[-lookback:]
   
    tot = np.stack((feature),axis=0).reshape(1,-1,1)
    Tot = scaler.fit_transform(tot[0])
    Tot = Tot.reshape(1,14,1)
    s = model.predict(Tot)
    S = scale.inverse_transform(s)[0][0]
    S = abs(S)


    feature.append(S)
    return feature


def updater_entrypoint(contexts, id, period, val_data):
    t = 0
    qt_cum_val = 0.0
    qo_cum_val = 0.0
    setpoint_glir = []
    qt_cum_val2 = 0.0
    qo_cum_val2 = 0.0
    setpoint_glir2 = []


    n_past = 14
    forecasting = []


    pp=p
    qq=q
    rr=r
    ii=i
    jj=j
    kk=k


    """o = list(np.ones(14)*180) #GLIR
    p = list(np.ones(14)*10) #wc
    q = list(np.ones(14)*1327) #ch
    r = list(np.ones(14)*14275) #gor


    h = list(np.ones(14)*1515)
    i = list(np.ones(14)*20)
    j = list(np.ones(14)*1254)
    k = list(np.ones(14)*5245)"""


    scaler = StandardScaler()
    scale = StandardScaler().fit(y)
    #x = ['glir11','wc11','ch11','gor11']
    t=0
    value_features = []
    value_pred = []
    qt_tot = []
    qw_tot = []


    while 1:
        # ========================================== WELL DYNAMICS (f_Qt(GLIR)) ==========================================
        setpoint = contexts.contexts[114].store["h"].getValues(7031+1,count=1)[0] #GLIR
        setpoint_glir.append(setpoint)
        #print('nilai setpoint',setpoint_glir)


        setpoint2 = contexts.contexts[116].store["h"].getValues(7031+1,count=1)[0] #GLIR
        setpoint_glir2.append(setpoint2)
        #print('nilai setpoint2',setpoint_glir2)


        # ========================================== PREP of NN Model ==========================================


        if t < n_past:
            pm = NN_Model_Params(model_wc1, pp, 14, x[['wc11']])
            qm = NN_Model_Params(model_ch1, qq, 14, x[['ch11']])
            rm = NN_Model_Params(model_gor1, rr, 14, x[['gor11']])


            im = NN_Model_Params(model_wc2, ii, 14, x2[['wc22']])
            jm = NN_Model_Params(model_ch2, jj, 14, x2[['ch22']])
            km = NN_Model_Params(model_gor2, kk, 14, x2[['gor22']])


            pp.append(pm[-1] + pm[-1]*random.uniform(-40/100, 40/100))
            qq.append(qm[-1] + qm[-1]*random.uniform(-40/100, 40/100))
            rr.append(rm[-1] + rm[-1]*random.uniform(-40/100, 40/100))
            ii.append(im[-1] + im[-1]*random.uniform(-40/100, 40/100))
            jj.append(jm[-1] + jm[-1]*random.uniform(-40/100, 40/100))
            kk.append(km[-1] + km[-1]*random.uniform(-40/100, 40/100))


            #print(f"GOR = {len(kk)}, {kk}")
            qo_simulated = NN_Model(model, 'OA-11', o, p, q, r, 14, y)
            qo_simulated2 = NN_Model(model1, 'OA-12', h,i,j,k, 14, y2)


        else:
            #print(f"GLIR 1:{setpoint_glir} \nGLIR2:{setpoint_glir2}")
            pm = NN_Model_Params(model_wc1, pp, 14, x[['wc11']])
            qm = NN_Model_Params(model_ch1, qq, 14, x[['ch11']])
            rm = NN_Model_Params(model_gor1, rr, 14, x[['gor11']])


            im = NN_Model_Params(model_wc2, ii, 14, x2[['wc22']])
            jm = NN_Model_Params(model_ch2, jj, 14, x2[['ch22']])
            km = NN_Model_Params(model_gor2, kk, 14, x2[['gor22']])


            pp.append(pm[-1] + pm[-1]*random.uniform(-40/100, 40/100))
            qq.append(qm[-1] + qm[-1]*random.uniform(-40/100, 40/100))
            rr.append(rm[-1] + rm[-1]*random.uniform(-40/100, 40/100))
            ii.append(im[-1] + im[-1]*random.uniform(-40/100, 40/100))
            jj.append(jm[-1] + jm[-1]*random.uniform(-40/100, 40/100))
            kk.append(km[-1] + km[-1]*random.uniform(-40/100, 40/100))


            #print(f"GOR = {len(kk)}, {kk}")
            qo_simulated = NN_Model(model, 'OA-11', setpoint_glir, p, q, r, 14, y)
            qo_simulated2 = NN_Model(model1, 'OA-12', setpoint_glir2, i, j, k, 14, y2)


        #MODEL WATERCUT
        #wc_val = WC_dyn(t)[0]
        #wc_val2 = WC_dyn(t)[1]
        wc_val = p[-1]/100
        wc_val2 = i[-1]/100
        print('watercut1=',wc_val)


        #MULTI WELL WELL MODEL REGRESSION ABSOLUTE
        #qo_simulated = 1*WellDyn(setpoint, 8.42570632e-05, -1.76777331e-01,  2.36235576e+02, 5)*math.exp(-0.000655*t)
        #qo_simulated = abs(qo_simulated + qo_simulated*random.uniform(-10/100, 10/100))*math.exp(-0.000655*t)
        qo_simulated = qo_simulated + qo_simulated*random.uniform(-10/100, 10/100)
        qo_simulated = abs(qo_simulated)
        qt_simulated = qo_simulated/(1 - WC_dyn(t)[0]) #TF BASED MODELs


        #qo_simulated2 = 1*WellDyn(setpoint2, 8.42570632e-05, -1.76777331e-01,  2.36235576e+02, 5)*math.exp(-0.000455*t)
        #qo_simulated2 = abs(qo_simulated2 + qo_simulated2*random.uniform(-10/100, 10/100))*math.exp(-0.000455*t)
        qo_simulated2 = qo_simulated2 + qo_simulated2*random.uniform(-10/100, 10/100)
        qo_simulated2 = abs(qo_simulated2)
        qt_simulated2 = qo_simulated2/(1 - WC_dyn(t)[1]) #TF BASED MODEL"""


        qt_cum_val += qt_simulated
        qo_cum_val += qo_simulated
        qt_cum_val2 += qt_simulated2
        qo_cum_val2 += qo_simulated2


        dict_data = {113: [qt_simulated,wc_val,qt_cum_val,qo_simulated,qo_cum_val,qq[-1],rr[-1]],
                    114: setpoint,
                    115: [qt_simulated2,wc_val2,qt_cum_val2,qo_simulated2,qo_cum_val2,jj[-1],kk[-1]],
                    116: setpoint2,
                    }
        val_data[id] = dict_data[id]


        # ========================================== STORE to REGISTERS ==========================================
        contexts.update_context(id,val_data[id])        
        #print("=================================================================================================")
        t += 1
        sleep(int(period))


def main():


    # ========================================== INPUT to SLAVE ==========================================
    slave_ids = [113,114,115,116]
    input_data = [[0,0.5,0,0,0,0,0],1000,[0,0.5,0,0,0,0,0],2000]
    periods = [1,1,1,1]
    periods = [2,2,2,2]
    modbus_template_paths = ['VX1.xlsx','ABB.xlsx','VX1.xlsx','ABB.xlsx']
   
    # ========================================== LOOPING SLAVE IDs ==========================================
    modbus_templates = {}
    val_data = {}
   
    for idx, id in enumerate(slave_ids):        
        template = ModbusTemplate(modbus_template_paths[idx])
        val_data[int(id)] = input_data[idx]
        modbus_templates[int(id)] = template
       
        logging.info(
            "simulating slave id {} with init. condition {} from {}".format(
                id, input_data[idx], modbus_template_paths[idx]
            )
        )
   
    print(template)
    print(f"TYPE MODTEMP: {modbus_templates}")
    print(f"TYPE VALDATA: {val_data}")
    # ========================================== SETTING UP CONTEXT ==========================================
    slave_contexts = SlaveContexts(slave_ids,modbus_templates,val_data)
    store = ModbusServerContext(slaves=slave_contexts.contexts, single=False)


    print(f"CTX: {slave_contexts}")
    print(f"STORE: {store}\n \n")
    # ========================================== INITIAL CONDITION ==========================================
    for i in slave_ids:
        slave_contexts.update_context(i, val_data[i])
   
    #slave_contexts.contexts[113].store["i"].setValues(int(entry.register_offset) + 1, val_data) #coba pake enumerate dan loop nanti
    #slave_contexts.contexts[114].store["h"].setValues(int(entry.register_offset) + 1, val_data)


    # ========================================== THREADING EACH SLAVE IDs ==========================================
    for idx, id in enumerate(slave_ids):
        updater = threading.Thread(
            target=updater_entrypoint, args=(
                slave_contexts,int(id), periods[idx], val_data)
        )


        updater.daemon = True
        updater.start()


    StartSerialServer(
        store, framer=ModbusRtuFramer, port='COM6', timeout=0.05, baudrate=9600
    )




if __name__ == "__main__":
    main()



