from Attack import Attack
from MNISTpart import ControlModel
from MNISTpart import MNISTdata
import torch

controlModel = ControlModel.ControlModel()
testloader = MNISTdata.MNISTdata().testloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# correct = 0
# total = 0
# for images,labels in testloader:
#   images, labels = images.to(device), labels.to(device)
#   correct = correct + torch.sum(torch.argmax(model(images),1) == labels)
#   total = total + labels.shape[0]
#
#
# print('Test Accuracy: %2.2f %%' % ((100.0 * correct) / total))
attack = Attack.Attack(controlModel.model, controlModel.loss_function, controlModel.optimiser)
res = attack.JSMA(testloader)
print(res)
