import yfinance as yf
import tensorflow as tf
import datetime
import numpy as np
import statistics as stats

ticker = "^NDX"
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(55)

lb = 60
lf = 12

stock_data = yf.download(ticker, start=start_date, end=end_date, interval='5m')
vals = stock_data['Open'].values
relative_rets = [[0 for j in range(lb)] for i in range(lb, len(vals) - lf)]

for i in range(lb, len(vals) - lf):
    for j in range(lb):
        relative_rets[i - lb][j] = vals[i] - vals[i - lb + j]
        relative_rets[i - lb][j] = relative_rets[i - lb][j]/vals[i - lb + j]

means = [stats.mean(rel) for rel in relative_rets]
devs = [stats.stdev(rel) for rel in relative_rets]

up_mom = [0 for i in range(lb, len(vals) - lf)]
down_mom = [0 for i in range(lb, len(vals) - lf)]

for i in range(lb, len(vals) - lf):
    for j in range(1, lf):
        r = vals[i + j] - vals[i]
        r = r/vals[i]

        if r > 0.005:
            up_mom[i - lb] = 1
        elif r < -0.005:
            down_mom[i - lb] = 1

n = len(relative_rets)
n = 4*(n//5)

train_rets = np.array(relative_rets[:n])
train_means = np.array(means[:n])
tran_devs = np.array(devs[:n])
train_up = np.array(up_mom[:n])
train_down = np.array(down_mom[:n])

test_rets = np.array(relative_rets[n:])
test_up = np.array(up_mom[n:])
test_down = np.array(down_mom[n:])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(2)
])

model2.compile(optimizer="adam",
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

model.fit(train_rets, train_up, epochs=25)
test_loss_up, test_acc_up = model.evaluate(test_rets, test_up, verbose=2)

rel_rn = [[0 for j in range(lb)] for i in range(lf)]

for i in range(len(vals) - lf, len(vals)):
    for j in range(lb):
        rel_rn[i - len(vals) + lf][j] = vals[i] - vals[i - lb + j]
        rel_rn[i - len(vals) + lf][j] = rel_rn[i - len(vals) + lf][j]/vals[i - lb + j]

probs_up = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
p_up = probs_up.predict(rel_rn)

model2.fit(train_rets, train_down, epochs=25)
test_loss_down, test_acc_down = model2.evaluate(test_rets, test_up, verbose=2)

probs_down = tf.keras.Sequential([model2, tf.keras.layers.Softmax()])
p_down = probs_down.predict(rel_rn)

print('\nTest accuracy up:', test_acc_up)
print(p_up)
print('\nTest accuracy down:', test_acc_down)
print(p_down)
