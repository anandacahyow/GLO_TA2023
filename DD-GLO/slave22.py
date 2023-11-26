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

logging.basicConfig(level=logging.INFO)

model = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel\RNN_model_resolved100') #BAGUSS
#model = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel\RNN_model_resolved_bagus_0.6r1')
#model = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_new\RNN_qo_type6_100+20')
model = load_model(r'C:\Users\ASUS\Documents\AllThingsPython\GLO_NN\NNModel_new\RNN_qo_type7_100+14')
print(f"\n model: <{model}> has been loaded\n")
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
            input_zero = np.ones((n_past-(t+1),))*2000
            input_zero = input_zero.tolist()

            input_totall = setpoint_glir + input_zero
            input_total = np.array(input_totall)
            input_total = np.reshape(input_total,(1,n_past,1))

            forecastt = model.predict(input_total) #shape = (n, 1) where n is the n_days_for_forecast
            forecastt = forecastt.tolist()[0][0]
            forecasting.append(forecastt)
            qo_simulated = forecastt
            qo_simulated2  =qo_simulated*0.75
            print('VALUE',input_total,'type',np.shape(input_total))
        else:
            input_totall = setpoint_glir[-n_past:]
            input_total = np.array(input_totall)
            input_total = np.reshape(input_total,(1,n_past,1))

            forecastt = model.predict(input_total) #shape = (n, 1) where n is the n_days_for_forecast
            forecastt = forecastt.tolist()[0][0]
            forecasting.append(forecastt)
            qo_simulated = forecastt
            qo_simulated2  =qo_simulated*0.75
            #print('VALUE',input_total,'type',np.shape(input_total))

        #MODEL WATERCUT
        wc_val = WC_dyn(t)[0]
        wc_val2 = WC_dyn(t)[1]

        #MULTI WELL WELL MODEL REGRESSION ABSOLUTE
        #qo_simulated = 1*WellDyn(setpoint, 8.42570632e-05, -1.76777331e-01,  2.36235576e+02, 5)*math.exp(-0.000655*t)
        qo_simulated = abs(qo_simulated + qo_simulated*random.uniform(-10/100, 10/100))
        qt_simulated = qo_simulated/(1 - WC_dyn(t)[0]) #TF BASED MODELs

        #qo_simulated2 = 1*WellDyn(setpoint2, 8.42570632e-05, -1.76777331e-01,  2.36235576e+02, 5)*math.exp(-0.000455*t)
        qo_simulated2 = abs(qo_simulated2 + qo_simulated2*random.uniform(-10/100, 10/100))
        qt_simulated2 = qo_simulated2/(1 - WC_dyn(t)[1]) #TF BASED MODEL"""

        qt_cum_val += qt_simulated
        qo_cum_val += qo_simulated
        qt_cum_val2 += qt_simulated2
        qo_cum_val2 += qo_simulated2

        dict_data = {113: [qt_simulated,wc_val,qt_cum_val,qo_simulated,qo_cum_val],
                    114: setpoint,
                    115: [qt_simulated2,wc_val2,qt_cum_val2,qo_simulated2,qo_cum_val2],
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
    input_data = [[0,0.5,0,0,0],2000,[0,0.5,0,0,0],2000]
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
