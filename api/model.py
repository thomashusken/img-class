from torchvision import models
import json
import numpy as np
import torch
from collections import OrderedDict
from operator import itemgetter
import os

def return_top_5(processed_image):
    # inception = models.inception_v3(pretrained=True)
    inception = models.inception_v3()
    inception.load_state_dict(torch.load("data/inception_v3_google-1a9a5a14.pth"))
    inception.eval()
    result = inception(processed_image)

    #load imagenet classes
    class_idx = json.load(open('data/imagenet_class_index.json'))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
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