import keras.models
import matplotlib.pyplot as plt
import numpy as np

import stock_parser as sp
import model_builder as mb
from checkers import *
from model_builder import ThresholdCallbackAcc
from interpretation import integrated_gradients, multiple_integrated_gradients

ticker = "^OMX"
interval = "5m"
back_date = 59
look_forward = 144
lookback = 10*look_forward
delta = 0.005
inputs = ['Open']
load = False

rel_rn, rel_rets, open_vals = sp.get_relative_stock_data(ticker,
                                                         backdate=back_date,
                                                         lb=lookback,
                                                         lf=look_forward,
                                                         interval=interval)

# histo = ret_interval(open_vals, look_forward)

mom = sp.single_mom_list(open_vals, lb=lookback, lf=look_forward, delta=delta)

up = 0
down = 0
for i in range(len(mom[:-look_forward])):
    if mom[i] == 2:
        up += 1
    elif mom[i] == 0:
        down += 1

up = up / len(mom[:-look_forward])
down = down / len(mom[:-look_forward])

print("Upward Momentum (with LF):", up)
print("Downward Momentum (with LF):", down)
print("No Momentum:", 1-(up + down))

"""
up_mom, down_mom = sp.momentum_check(vals, lb=lookback, lf=look_forward, delta=delta)

up_model, up_data = mb.generate_tuned_model("s_" + ticker + "_" + interval,
                                            "up_mom",
                                            rel_rets[:-look_forward],
                                            up_mom[:-look_forward])
down_model, down_data = mb.generate_tuned_model("s_" + ticker + "_" + interval,
                                                "down_mom",
                                                rel_rets[:-look_forward],
                                                down_mom[:-look_forward])


val_up = up_model.predict(rel_rets[-look_forward:])
val_down = down_model.predict(rel_rets[-look_forward:])

length = len(rel_rets[-look_forward:])

up_check, down_check, false_pos_up, false_neg_up, false_pos_down, false_neg_down = bayesian(val_up,
                                                                                            val_down,
                                                                                            up_mom,
                                                                                            down_mom,
                                                                                            length,
                                                                                            look_forward)

up_model.summary()
print("Upward Val_accuracy:", up_data[3])
print("Upward ratio of fails:", 1-sum(up_mom[:-look_forward])/len(up_mom[:-look_forward]))

print("\n")
down_model.summary()
print("Downward Val_accuracy:", down_data[3])
print("Downward ratio of fails", 1-sum(down_mom[:-look_forward])/len(down_mom[:-look_forward]))

print(1 - sum(up_mom[-look_forward:])/len(up_mom[-look_forward:]))
print(up_check[-1])
print(1 - sum(down_mom[-look_forward:])/len(down_mom[-look_forward:]))
print(down_check[-1])

fig = plt.plot(range(len(up_check)), up_check, label="UpMom")
plt.plot(range(len(up_check)), false_pos_up, label="FalsePosUp")
plt.plot(range(len(up_check)), false_neg_up, label="FalseNegUp")
plt.plot(range(len(down_check)), down_check, label="DownMom")
plt.legend()
plt.show()

update = input("Enter 1 to Update Model with these predictions and predict the future: ")
if update == "1":
    up_model.fit(rel_rets[-look_forward:], up_mom[-look_forward:], verbose=0)
    down_model.fit(rel_rets[-look_forward:], down_mom[-look_forward:], verbose=0)

    p_up = up_model.predict(rel_rn)
    p_down = down_model.predict(rel_rn)

    print(p_down[-10:])
    print(p_up[-10:])
else:
    print("Okili Dokili")
    
"""
folder = "solo" + \
         "_" + interval + \
         "_" + str(delta) + \
         "_" + str(back_date) + \
         "_" + str(lookback) + \
         "_" + str(look_forward) + \
         "_in_" + str(len(inputs))

model = None
data = None

if load:
    model = keras.models.load_model("C:\\Users\\coeno\\PycharmProjects\\MLStonks\\Models\\" + ticker + "\\" + folder)
    data = [0, 0, 0, 0]
else:
    model, data = mb.generate_tuned_model("s_" + ticker,
                                          folder,
                                          rel_rets[:-look_forward],
                                          mom[:-look_forward])

val_mod = model.predict(rel_rets[-look_forward:])
length = len(rel_rets[-look_forward:])


rel_rets_sub = list()
targets = list()

check, f_neg, f_not, f_pos = ternary_bayesian(val_mod, mom, length, look_forward)
model.summary()

print("Model Val_Accuracy:", data[3])
print('\n')

print("Model Unseen Check:", check[-1])
print("Model False Pos:", f_pos[-1])
print("Model False Neg:", f_neg[-1])
print("Model False Nothing:", f_not[-1])

fig = plt.plot(range(len(check)), check, label="Sanity")
plt.plot(range(len(check)), np.sum(f_pos, axis=1), label="False Pos")
plt.plot(range(len(check)), np.sum(f_neg, axis=1), label="False Neg")
plt.plot(range(len(check)), np.sum(f_not, axis=1), label="False Nothing")
plt.legend()

plt.show()

update = input("Please enter '1' in order to update the model: ")

if update == "1":
    callback = ThresholdCallbackAcc(0.95)
    model.fit(rel_rets[-look_forward:], mom[-look_forward:], epochs=20, verbose=1, callbacks=[callback])
    pred = model.predict(rel_rn)

    s = 0
    for i in pred:
        s += i.argmax()

    s = s / len(pred)
    print(s)

    print(pred[-5:])

    x = range(1, len(pred) + 1)
    neg = list()
    noth = list()
    pos = list()

    for p in pred:
        neg.append(p[0])
        noth.append(p[1])
        pos.append(p[2])

    fig, axs = plt.subplots()
    axs.plot(x, neg, label="Decrease")
    axs.plot(x, noth, label="Nothing")
    axs.plot(x, pos, label="Increase")
    axs.legend()
    plt.show()

    update = input("Please enter '1' in order to save the model: ")
    if update == "1":
        print("Saving!")
        model.save("C:\\Users\\coeno\\PycharmProjects\\MLStonks\\Models\\" + ticker + "\\" + folder, ticker + ".keras")
else:
    print("Okilidokili")
