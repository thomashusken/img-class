from torchvision import models
import json
import numpy as np



def return_top_5(processed_image):
    inception = models.inception_v3(pretrained=True)
    inception.eval()
    result = inception(processed_image)

    #load imagenet classes
    class_idx = json.load(open('data/imagenet_class_index.json'))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    result_idx = result.sort()[1][0][-5:]

    #exponentiate and get probabilities
    exps = np.exp(result.detach().numpy()[0])
    exps_sum = np.sum(exps)
    softmax = [j / exps_sum for j in exps]

    out = []
    for idx in result_idx:
        out.append((idx2label[idx], softmax[idx]))

    out = sorted(out, key=lambda x: x[1], reverse=True)

    return out