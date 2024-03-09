def should_you_invest_duo(up_model, down_model, cur_val, ratio=5):
    up_prediction = up_model.predict(cur_val, verbose=0)
    down_prediction = down_model.predict(cur_val, verbose=0)

    if up_prediction[0].argmax() == 1:
        if up_prediction[0][1]/down_prediction[0][1] > ratio:
            return 1

    if down_prediction[0].argmax() == 1:
        if down_prediction[0][1]/up_prediction[0][1] > ratio:
            return -1

    return 0


def should_you_invest_solo(model, cur_val, ratio=1):

    prediction = model.predict(cur_val, verbose=0)

    if prediction[0].argmax() == 0:
        try:
            if prediction[0][0]/prediction[0][2] > ratio:
                return -1
        except RuntimeWarning:
            return -1

    if prediction[0].argmax() == 2:
        try:
            if prediction[0][2]/prediction[0][0] > ratio:
                return 1
        except RuntimeWarning:
            return 1

    return 0


def schizo_strategy(model, cur_val, ratio=1):
    prediction = model.predict(cur_val, verbose=0)

    if prediction[0].argmax() == 0:
        try:
            if prediction[0][0] / prediction[0][2] > ratio:
                return 1
        except RuntimeWarning:
            return 1

    if prediction[0].argmax() == 2:
        try:
            if prediction[0][2] / prediction[0][0] > ratio:
                return -1
        except RuntimeWarning:
            return -1

    return 0
