from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


class MNISTdata():
    def __init__(self):
        self.trainloader, self.testloader = self.getLoader()

    def getLoader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])

        trainset = MNIST('.', train=True, download=True, transform=transform)
        testset = MNIST('.', train=False, download=True, transform=transform)

        trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
        testloader = DataLoader(testset, batch_size=128, shuffle=True)

        return trainloader, testloader
