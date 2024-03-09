from stock_parser import get_relative_stock_data, ret_histograms
import numpy as np
import matplotlib.pyplot as plt
from checkers import ret_histogram_skeleton
from model_builder import histogram_model, toy_histogram_model, generate_tuned_histogram

ticker = "^OMX"
interval = "5m"
back_date = 59
look_forward = 144
lookback = 15*look_forward
delta = 0.005
inputs = ['Open']
load = False
batch_size = 50

folder = "solo" + \
         "_" + interval + \
         "_" + str(delta) + \
         "_" + str(back_date) + \
         "_" + str(lookback) + \
         "_" + str(look_forward) + \
         "_in_" + str(len(inputs))

rel_rn, rel_rets, open_vals = get_relative_stock_data(ticker,
                                                      backdate=back_date,
                                                      lb=lookback,
                                                      lf=look_forward,
                                                      interval=interval)

total_rets = rel_rets
total_rets.extend(rel_rn)

width, edges = ret_histogram_skeleton(total_rets, True)

ret_hists, target_hists, rn_hists = ret_histograms(open_vals,
                                                   total_rets,
                                                   edges,
                                                   lb=lookback,
                                                   lf=look_forward,
                                                   batch_size=batch_size)

model, data = generate_tuned_histogram("h_" + ticker,
                                       folder,
                                       ret_hists[-look_forward:],
                                       target_hists[-look_forward:],
                                       )

print("Checking Sanity")
# sanity = model.evaluate(ret_hists[:-look_forward],
#                        target_hists[:-look_forward]
#                        )

showcase = model.predict(ret_hists[-2:])

fig, axes = plt.subplots(1, 3)
axes[0].bar(edges[:-1], showcase[-1], width=0.8*width, label="Prediction")
axes[0].legend()
axes[1].bar(edges[:-1], showcase[-1] * np.diff(edges), width=0.8*width, label="Prediction with Diff")
axes[1].legend()
axes[2].bar(edges[:-1], target_hists[-2] * np.diff(edges), width=0.8*width, label="Target")
axes[2].legend()
plt.show()

print("Predicting")
predict = model.predict(rn_hists)

_ = plt.bar(edges[:-1], predict[-1], width=0.8*width, label="Future Returns")
plt.legend()
plt.show()
