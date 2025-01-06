import glob
import torch

def Corrupt_mnist():

        train_paths = zip(
                    sorted(glob.glob("data/corruptmnist_v1/train_images_*.pt")), 
                    sorted(glob.glob("data/corruptmnist_v1/train_target_*.pt"))
                    )
        train = ((torch.load(x), torch.load(y)) for x, y in train_paths)

        test_paths = zip(
                    sorted(glob.glob("data/corruptmnist_v1/test_images.pt")), 
                    sorted(glob.glob("data/corruptmnist_v1/test_target.pt"))
                    )
        test = ((torch.load(x), torch.load(y)) for x, y in test_paths)

        return train, test
