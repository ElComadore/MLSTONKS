import keras
import stock_parser as sp
import model_builder as mb
import matplotlib.pyplot as plt
import invest_strategy as i_s
from checkers import ternary_bayesian


ticker = "^OMX"
interval = "5m"
back_date = 59
look_forward = 144
lookback = 10*look_forward
delta = 0.005
inputs = ['Open']
load = True

rel_rn, total_rel_rets, open_vals = sp.get_relative_stock_data(ticker,
                                                               backdate=back_date,
                                                               lb=lookback,
                                                               lf=look_forward,
                                                               interval=interval,
                                                               )

total_mom = sp.single_mom_list(open_vals, lb=lookback, lf=look_forward, delta=delta)
amount = int(0.5*len(total_rel_rets))
i = amount

training_rel_rets = total_rel_rets[:amount]
testing_rel_rets = total_rel_rets[amount:]
training_mom = total_mom[:amount]
test_mom = total_mom[amount:]

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
    model, data = mb.generate_tuned_model("t_" + ticker,
                                          folder,
                                          total_rel_rets[:-look_forward],
                                          total_mom[:-look_forward])

model.summary()

pause = input("Enter to continue")

print("Starting Invest")
num_trades = 0
successful_trades = 0
total_wealth = [1]
refit = -1
invest_val = 0
long = None
invested = False
ratio = 100
# strategy = i_s.schizo_strategy
strategy = i_s.should_you_invest_solo

while i < len(total_rel_rets):
    wealth_changed = False
    if invested:
        ret = open_vals[i] - invest_val
        ret = ret / invest_val

        if abs(ret) >= delta:
            invested = False

            if (long and ret > 1) or (not long and ret < 1):
                total_wealth.append(total_wealth[-1] * (1 + delta))
                print(f"Divesting at {i} with gain {delta}: {total_wealth[-1]}")
                successful_trades += 1
            else:
                total_wealth.append(total_wealth[-1] * (1 - delta))
                print(f"Divesting at {i} with loss {-delta}: {total_wealth[-1]}")

            wealth_changed = True
        else:
            total_wealth.append(total_wealth[-1])

    if not invested:
        should = strategy(model, [total_rel_rets[i]], ratio)

        if should == 0:
            pass
        elif should == 1:
            long = True
            invested = True
            invest_val = open_vals[i]
            print(f"Taking Long Position at {i}")
            num_trades += 1
        else:
            long = False
            invested = True
            invest_val = open_vals[i]
            print(f"Taking Short position at {i}")
            num_trades += 1

        if not wealth_changed:
            total_wealth.append(total_wealth[-1])

    if refit != -1:
        if (i - amount) % refit == 0:
            model.fit(total_rel_rets[i-refit:i], total_mom[i - refit:i])

    i += 1

rets = list()
for i in range(amount, len(total_rel_rets)):
    tmp = open_vals[i]/open_vals[amount]
    rets. append(tmp)

fig = plt.plot(range(len(total_wealth)), total_wealth)
plt.plot(range(len(rets)), rets, 'g')
print(total_wealth[-1])

val_mod = model.predict(testing_rel_rets)
check, f_neg, f_not, f_pos = ternary_bayesian(val_mod, test_mom, length=len(val_mod), look_forward=look_forward)

up = 0
down = 0
for i in range(len(training_mom)):
    if training_mom[i] == 2:
        up += 1
    elif training_mom[i] == 0:
        down += 1

up = up / len(training_mom)
down = down / len(training_mom)

print("Upward Momentum (with LF):", up)
print("Downward Momentum (with LF):", down)
print("No Momentum:", 1-(up + down))

print('\n')

print("Model Unseen Check:", check[-1])
print("Model False Pos:", f_pos[-1])
print("Model False Neg:", f_neg[-1])
print("Model False Nothing:", f_not[-1])

plt.show()

save = input("Enter '1'to save the model: ")
if save != '1':
    exit(-1)

print("Saving!")
model.save("C:\\Users\\coeno\\PycharmProjects\\MLStonks\\Models\\" + ticker + "\\" + folder)
