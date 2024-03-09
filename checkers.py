import numpy as np
import matplotlib.pyplot as plt


def binary_bayesian(pred, real, length, look_forward):
    check = [0 for _ in range(length)]
    false_pos = [0 for _ in range(length)]
    false_neg = [0 for _ in range(length)]

    if pred[0].argmax() == real[-look_forward]:
        check[0] = 1
    else:
        if pred[0].argmax() == 0:
            false_neg[0] = 1
        else:
            false_pos[0] = 1

    for i in range(1, length):
        if pred[i].argmax() == real[-look_forward + i]:
            check[i] = check[i - 1] + 1
            false_neg[i] = false_neg[i - 1]
            false_pos[i] = false_pos[i - 1]
        else:
            check[i] = check[i - 1]
            if pred[i].argmax() == 0:
                false_neg[i] = false_neg[i - 1] + 1
                false_pos[i] = false_pos[i - 1]
            else:
                false_pos[i] = false_pos[i - 1] + 1
                false_neg[i] = false_neg[i - 1]

    for i in range(1, len(check)):
        check[i] = check[i] / (i + 1)

        false_pos[i] = false_pos[i] / (i + 1)
        false_neg[i] = false_neg[i] / (i + 1)

    return check, false_pos, false_neg


def ternary_bayesian(pred, real, length, look_forward):
    check = [0 for _ in range(length)]
    false_nothing = [[0 for _ in range(3)] for _ in range(length)]
    false_pos = [[0 for _ in range(3)] for _ in range(length)]
    false_neg = [[0 for _ in range(3)] for _ in range(length)]

    if pred[0].argmax() == real[-look_forward]:
        check[0] = 1
    else:
        if pred[0].argmax() == 0:
            false_neg[0][real[-look_forward]] = 1
        elif pred[0].argmax() == 1:
            false_nothing[0][real[-look_forward]] = 1
        else:
            false_pos[0][real[-look_forward]] = 1

    for i in range(1, length):
        if pred[i].argmax() == real[-look_forward + i]:
            check[i] = check[i - 1] + 1
            for j in range(3):
                false_neg[i][j] = false_neg[i - 1][j]
                false_pos[i][j] = false_pos[i - 1][j]
                false_nothing[i][j] = false_nothing[i - 1][j]

        else:
            check[i] = check[i - 1]
            if pred[i].argmax() == 0:
                false_neg[i][real[-look_forward + i]] = false_neg[i - 1][real[-look_forward + i]] + 1

                for j in range(3):
                    false_pos[i][j] = false_pos[i - 1][j]
                    false_nothing[i][j] = false_nothing[i - 1][j]

                    if j != real[-look_forward + i]:
                        false_neg[i][j] = false_neg[i - 1][j]

            elif pred[i].argmax() == 1:
                false_nothing[i][real[-look_forward + i]] = false_nothing[i - 1][real[-look_forward + i]] + 1

                for j in range(3):
                    false_neg[i][j] = false_neg[i - 1][j]
                    false_pos[i][j] = false_pos[i - 1][j]

                    if j != real[-look_forward + i]:
                        false_nothing[i][j] = false_nothing[i - 1][j]

            else:
                false_pos[i][real[-look_forward + i]] = false_pos[i - 1][real[-look_forward + i]] + 1

                for j in range(3):
                    false_neg[i][j] = false_neg[i - 1][j]
                    false_nothing[i][j] = false_nothing[i - 1][j]

                    if j != real[-look_forward + i]:
                        false_pos[i][j] = false_pos[i - 1][j]

    check = np.divide(check, range(1, len(check) + 1))

    for i in range(len(false_nothing)):
        false_neg[i] = np.divide(false_neg[i], (i + 1))
        false_pos[i] = np.divide(false_pos[i], (i + 1))
        false_nothing[i] = np.divide(false_nothing[i], (i + 1))

    return check, false_neg, false_nothing, false_pos


def ret_interval(vals, look_forward):
    ret_hist = list()

    for i in range(len(vals) - look_forward):
        for j in range(1, look_forward):
            ret = vals[i + j] - vals[i]
            ret = ret / vals[i]

            ret_hist.append(ret)

    ret_hist, edges = np.histogram(ret_hist, bins=750, density=True)
    com_sum = 0
    inter = list()

    for i in range(len(ret_hist)):
        com_sum += ret_hist[i]/ret_hist.sum()

        if com_sum > 0.05 and len(inter) == 0:
            inter.append(edges[i])

        elif com_sum > 0.95 and len(inter) == 1:
            inter.append(edges[i])
            break

    print(inter)

    fig = plt.stairs(values=ret_hist, edges=edges)
    plt.show()

    return ret_hist


def ret_histogram_skeleton(rets: list[list], show=False):

    smallest = 0
    largest = 0

    for i in range(len(rets)):
        for j in range(len(rets[0])):
            smallest = min(smallest, rets[i][j])
            largest = max(largest, rets[i][j])

    _, bins = np.histogram(rets[0], bins="sqrt")

    width = bins[1] - bins[0]
    edges = np.arange(smallest-width, largest+width, width)

    if show:
        hist, _ = np.histogram(np.array(rets).flatten(), bins=edges, density=True)
        plt.bar(edges[:-1], hist * np.diff(edges), width=0.9*width)
        plt.show()

    return width, edges
