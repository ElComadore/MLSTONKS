import numpy as np
from model_builder import generate_tuned_func, functional_model
from stock_parser import get_relative_dict, single_mom
from checkers import ternary_bayesian
import matplotlib.pyplot as plt


ticker = "MSFT"
interval = "1d"
back_date = 15000
look_forward = 30
lookback = 8*look_forward
delta = 0.05
inputs = ['Open']
load = False

folder = "solo" + \
         "_" + interval + \
         "_" + str(delta) + \
         "_" + str(back_date) + \
         "_" + str(lookback) + \
         "_" + str(look_forward) + \
         "_in_" + str(len(inputs))

vals, rel_rets, rel_rn = get_relative_dict(ticker,
                                           backdate=back_date,
                                           lf=look_forward,
                                           lb=lookback,
                                           interval=interval)

training = {"Open": rel_rets['Open'][:-look_forward],
            "High": rel_rets['High'][:-look_forward],
            "Low": rel_rets['Low'][:-look_forward]}

n = int(0.7 * len(training['Open']))

open_training = training['Open'][:n]
high_training = training['High'][:n]
low_training = training['Low'][:n]

open_valid = training['Open'][n:]
high_valid = training['High'][n:]
low_valid = training['Low'][n:]

rest_rel_rets = {"Open": rel_rets['Open'][-look_forward:],
                 "High": rel_rets['High'][-look_forward:],
                 "Low": rel_rets['Low'][-look_forward:]}


mom = single_mom(vals['Open'], lf=look_forward, lb=lookback, delta=delta)

"""
model = functional_model()
inputs = [open_training, high_training, low_training]
# {"Open": open_training, "High": high_training, "Low": low_training}, mom[:n]

model.fit(training,
          mom[:-look_forward],
          epochs=10,
          validation_split=0.8,
          verbose=1)
"""

model = generate_tuned_func("f_"+ticker,
                            folder,
                            training,
                            mom[:-look_forward])

val_mod = model.predict(rest_rel_rets)
length = len(rest_rel_rets['Open'])

check, f_neg, f_not, f_pos = ternary_bayesian(val_mod, mom, length, look_forward)
model.summary()

up = 0
down = 0
for i in range(len(mom[:-look_forward])):
    if mom[i] == 2:
        up += 1
    elif mom[i] == 0:
        down += 1

up = up / len(mom[:-look_forward])
down = down / len(mom[:-look_forward])


# print("Model Val_Accuracy:", data[3])
print("Upward Momentum (with LF):", up)
print("Downward Momentum (with LF):", down)
print("No Momentum:", 1-(up + down))
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

model.fit(rest_rel_rets, mom[-look_forward:])
