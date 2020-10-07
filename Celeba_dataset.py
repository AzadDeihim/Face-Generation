import os

import numpy as np
from PIL import Image as PILImage
from torch.utils import data
from torchvision import transforms


class CelebaDataset(data.Dataset):

    def __init__(self, **kwargs):
        self.image_dir = kwargs['dataset_dir'] + '/img_align_celeba'
        self.image_size = kwargs['img_size']
        self.list_image_names = os.listdir(self.image_dir)

    # Size of the dataset
    def __len__(self):
        return len(self.list_image_names)

    # Return one image
    def __getitem__(self, index):
        X = self.load_image(index)
        return X, index

    # Normalise image and converts to PyTorch tensor
    @staticmethod
    def transform_image(image, image_size):
        h_, w_ = image_size[0], image_size[1]
        im_size = tuple([h_, w_])

        # mean values for RGB
        t_ = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                 std=[1, 1, 1])
        ])

        image = t_(image)
        # TODO: check normalisation is between -1 and 1
        return image

    # load one image
    # index: index in the list of images
    def load_image(self, index):
        image = np.array(PILImage.open(os.path.join(self.image_dir, self.list_image_names[index])))
        image = self.transform_image(image, self.image_size)
        return image
