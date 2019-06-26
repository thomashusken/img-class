import json
import numpy as np
import torch
from collections import OrderedDict
from operator import itemgetter
import os

def return_top_5(processed_image, model, idx2label):
    result = model(processed_image)
    result_idx = result.sort()[1][0][-5:]

    #exponentiate and get probabilities
    exps = np.exp(result.detach().numpy()[0])
    exps_sum = np.sum(exps)
    softmax = [np.round((j / exps_sum)*100, 2) for j in exps]

    out = []
    for idx in result_idx:
        out.append((idx2label[idx], softmax[idx]))

#    out = {k: v for k, v in dict(out).items()}
    result = OrderedDict(sorted(dict(out).items(), key=itemgetter(1), reverse=True))

    return result