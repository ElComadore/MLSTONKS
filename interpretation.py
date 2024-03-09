import tensorflow as tf
import numpy as np


def interpolate_rets(baseline, rel_rets, alphas):
    delta = np.subtract(rel_rets, baseline)

    rel_rets_2 = list()
    for alpha in alphas:
        tmp = baseline + alpha*delta
        rel_rets_2.append(np.transpose(tmp).tolist())

    return rel_rets_2


def compute_grads(model, rel_rets, target_class):
    probs = model.predict(rel_rets)
    gradients = list()

    for i in range(len(rel_rets) - 1):
        tmp = probs[i + 1][target_class] - probs[i][target_class]
        tmp = tmp / (rel_rets[i + 1][target_class] - rel_rets[i][target_class])

        gradients.append(tmp)

    return gradients


def one_batch(model, baseline, rel_rets, alpha_batch, target_class):
    interpolated_path = interpolate_rets(baseline=baseline, rel_rets=rel_rets, alphas=alpha_batch)

    gradient_batch = compute_grads(model=model, rel_rets=interpolated_path, target_class=target_class)

    return gradient_batch


def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    int_grads = np.mean(grads)

    return int_grads


def integrated_gradients(model, baseline, rel_rets, target_class, m_steps=50, batch_size=32):

    # Generate interpolation
    alphas = np.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Collect batches
    gradient_batches = []

    # Iterate over the alphas

    for alpha in range(0, len(alphas), batch_size):
        from_ = alpha
        to = np.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        gradient_batch = one_batch(model, baseline, rel_rets, alpha_batch, target_class)
        gradient_batches.append(gradient_batch)

    total_gradients = np.concatenate(gradient_batches, axis=0)
    avg_gradients = integral_approximation(gradients=total_gradients)

    int_grads = np.subtract(rel_rets, baseline) * avg_gradients

    return int_grads


def multiple_integrated_gradients(model, baseline, rel_rets_list, target_classes, m_steps=50, batch_size=32):
    all_grads = list()

    for i in range(len(rel_rets_list)):
        tmp = integrated_gradients(model, baseline, rel_rets_list[i], target_classes[i], m_steps, batch_size)
        all_grads.append(tmp)

    return all_grads


def plt_integrated_grads(model, rel_rets_list, target_classes, baseline=None, m_steps=50, batch_size=32):
    import matplotlib.pyplot as plt

    if isinstance(rel_rets_list[0], list):
        if baseline is None:
            baseline = list()

            for i in range(len(rel_rets_list[0])):
                baseline.append(0)

        all_grads = multiple_integrated_gradients(model=model,
                                                  baseline=baseline,
                                                  rel_rets_list=rel_rets_list,
                                                  target_classes=target_classes,
                                                  m_steps=m_steps,
                                                  batch_size=batch_size)

        x = np.divide(np.arange(0, len(all_grads[0])), 12)

        fig = plt.plot(np.flip(x), all_grads[0], label=f"Direction: {target_classes[0]}")

        for i in range(1,len(target_classes)):
            plt.plot(np.flip(x), all_grads[i], label=f"Direction: {target_classes[i]}")

        plt.legend()
        plt.show()

    else:
        if baseline is None:
            baseline = list()

            for i in range(len(rel_rets_list)):
                baseline.append(0)

        int_grads = integrated_gradients(model=model,
                                         baseline=baseline,
                                         rel_rets=rel_rets_list,
                                         target_class=target_classes,
                                         m_steps=m_steps,
                                         batch_size=batch_size)

        x = np.divide(np.arange(0, len(int_grads)), 12)

        fig = plt.plot(np.flip(x), int_grads, label=f"Direction: {target_classes}")

        plt.legend()
        plt.show()


