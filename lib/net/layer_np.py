from common import *

def np_metric_accuracy(predict, truth):
    truth = truth.reshape(-1)
    predict = predict.reshape(-1)

    correct = truth == predict
    correct = correct.mean()
    return correct

def np_loss_cross_entropy(probability, truth):
    batch_size = len(probability)
    truth = truth.reshape(-1).astype(np.int)

    p = probability[np.arange(batch_size), truth]
    loss = -np.log(np.clip(p, 1e-6, 1))
    loss = loss.mean()
    return loss