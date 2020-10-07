import os

import numpy as np
import torch.utils.data
import torchvision
from PIL import Image as PILImage
from scipy.stats import entropy
from torch.nn import functional as F
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device = %s' % device)

########################################################################################################################
# Define variables
########################################################################################################################
splits = 10
batch_size = 16
image_dir = './......'

########################################################################################################################
# Define dataset class
########################################################################################################################


class InceptionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.list_image_names = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.list_image_names)

    def __getitem__(self, index):
        image = np.array(PILImage.open(os.path.join(self.image_dir, self.list_image_names[index])))
        image = self.transform_image(image)
        return image, index

    @staticmethod
    def transform_image(image):
        # mean values for RGB
        t_ = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                                      std=[0.225, 0.224, 0.229])
                                 ])
        image = t_(image)
        image = F.interpolate(image.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=True).squeeze(0)
        return image


########################################################################################################################
# Create dataset and dataloader
########################################################################################################################
images_dataset = InceptionDataset(image_dir=image_dir)
num_images = len(images_dataset)

dataloader = torch.utils.data.DataLoader(images_dataset, batch_size=batch_size)

########################################################################################################################
# Load inception-v3
########################################################################################################################
print('Loading inception-v3 model')
# TODO: change model save location?
inception_model = torchvision.models.inception_v3(pretrained=True, transform_input=False, progress=True)
inception_model = inception_model.to(device)
inception_model.eval()

########################################################################################################################
# Evaluation loop
########################################################################################################################
# Create predictions vector of size (num_images, num_inception-v3_classes)
predictions = np.zeros((num_images, 1000))

print('Evaluating images using inception-v3')
for i, batch in enumerate(dataloader, start=0):
    if i % 100 == 0:
        print('Batch [%g/%g]' % (i, num_images // batch_size))
    image_batch = batch[0].to(device)
    batch_size_i = image_batch.size(0)

    with torch.no_grad():
        image_batch = inception_model(image_batch)

    predictions[i * batch_size: i * batch_size + batch_size_i] = F.softmax(image_batch, dim=1).detach().cpu().numpy()

########################################################################################################################
# Calculate mean Kullback–Leibler divergence
########################################################################################################################
print('Calculating mean Kullback–Leibler divergence')
split_scores = []
for k in range(splits):
    part = predictions[k * (num_images // splits): (k + 1) * (num_images // splits), :]
    py = np.mean(part, axis=0)
    scores = []
    for i in range(part.shape[0]):
        pyx = part[i, :]
        scores.append(entropy(pyx, py))
    split_scores.append(np.exp(np.mean(scores)))

inception_score_mean = np.mean(split_scores)
inception_score_std = np.std(split_scores)
print('Inception Score: (mean, std) = (%g, %g)' % (inception_score_mean, inception_score_std))

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Test dataset ==> # TODO: delete
########################################################################################################################


# class IgnoreLabelDataset(torch.utils.data.Dataset):
#     def __init__(self, orig):
#         self.orig = orig
#
#     def __getitem__(self, index):
#         image = self.orig[index][0]
#         image = F.interpolate(image.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=True).squeeze(0)
#         return image, index
#
#     def __len__(self):
#         return len(self.orig)
#
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
#
# cifar = dset.CIFAR10(root='data/', download=True,
#                      transform=transforms.Compose([
#                          transforms.Resize(32),
#                          transforms.ToTensor(),
#                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                      ])
#                      )
