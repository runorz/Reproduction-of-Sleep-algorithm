from tqdm import tqdm
import numpy as np
import torch

class Attack():
    def __init__(self, model, loss_function, optimiser):
        self.model = model
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def FGSM(self, testloader):
        n = torch.zeros(testloader.dataset.data.shape[0])
        j = 0
        with tqdm(total=testloader.dataset.data.shape[0]) as pbar:
            for images, labels in testloader:
                for i in range(images.shape[0]):
                    img = images[i]
                    img = img.reshape(1,-1)
                    img = img.to(self.device)
                    x = images[i]
                    x = x.reshape(1,-1)
                    x = x.to(self.device)
                    x.requires_grad = True
                    y = labels[i]
                    y = y.reshape(-1)
                    y = y.to(self.device)
                    temp = 0.0000
                    while torch.argmax(self.model(x.data)) == y:
                        x.grad = None
                        self.optimiser.zero_grad()
                        temp += 0.0001
                        out = self.model(x)
                        loss = self.loss_function(out, y)
                        loss.backward()
                        x.data = img.data + torch.sign(x.grad) * temp
                    n[j] = temp
                    j += 1
                    pbar.update(1)
        return torch.mean(n)

    def DeepFool(self, testloader, max_iter=100):
        n = torch.zeros(testloader.dataset.data.shape[0])
        j = 0
        with tqdm(total=testloader.dataset.data.shape[0]) as pbar:
            for images, labels in testloader:
                for i in range(images.shape[0]):
                    img = images[i]
                    img = img.reshape(1,-1)
                    img = img.to(self.device)
                    x = torch.clone(img)
                    x = x.to(self.device)
                    x.requires_grad = True

                    y = labels[i]
                    y = y.to(self.device)

                    w = torch.zeros(x.shape)
                    r_tot = torch.zeros(x.shape)
                    r_tot = r_tot.to(self.device)

                    out = torch.flatten(self.model(x))
                    k_i = torch.argmax(out)

                    loop = 0
                    while k_i == y and loop < max_iter:
                        perk = np.inf
                        out[k_i].backward(retain_graph=True)
                        grad_ori = x.grad

                        for k in range(out.shape[-1]):
                            if k != k_i:
                                x.grad = None
                                self.optimiser.zero_grad()

                                out[k].backward(retain_graph=True)
                                grad_cur = x.grad

                                w_k = grad_cur - grad_ori
                                f_k = out[k] - out[k_i]

                                perk_k = abs(f_k) / torch.norm(w_k)

                                if perk_k < perk:
                                    perk = perk_k
                                    w = w_k

                        r_i = perk * w / torch.norm(w)
                        r_tot += r_i

                        x.data = img + r_tot
                        x.grad = None
                        self.optimiser.zero_grad()
                        out = torch.flatten(self.model(x))
                        k_i = torch.argmax(out)
                        loop += 1

                    n[j] = torch.norm(r_tot) / x.shape[-1]
                    j += 1
                    pbar.update(1)
        return torch.mean(n)

    def JSMA(self,testloader):
        n = torch.zeros(10000)
        j = 0
        with tqdm(total=10000) as pbar:
            for images, labels in testloader:
                for i in range(images.shape[0]):
                    img = images[i]
                    img = img.to(self.device)

                    x = images[i]
                    x = x.reshape(1,-1)
                    x = x.to(self.device)
                    x.requires_grad = True
                    y = labels[i]
                    y = y.reshape(-1)
                    y = y.to(self.device)

                    cur_iter = 0

                    freq = torch.zeros(x.shape[-1])

                    while torch.argmax(self.model(x.data)) == y and cur_iter < 500:
                        x.grad = None
                        self.optimiser.zero_grad()
                        out = torch.flatten(self.model(x))

                        smap = torch.zeros(x.shape[-1])

                        out[y].backward(retain_graph=True)
                        y_grad =  torch.flatten(x.grad)

                        else_grad = torch.zeros(y_grad.shape)
                        else_grad = else_grad.to(self.device)
                        for index in range(10):
                            if index != y:
                                x.grad = None
                                self.optimiser.zero_grad()
                                out[index].backward(retain_graph=True)
                                else_grad += torch.flatten(x.grad)

                        for index in range(smap.shape[-1]):
                            if y_grad[index] > 0 or else_grad[index] < 0:
                                smap[index] = 0
                            else:
                                smap[index] = torch.abs(y_grad[index]) * else_grad[index]


                        max_value = 0
                        max_index = 0
                        second_max_value = 0
                        second_max_index = 0
                        for index in range(smap.shape[-1]):
                            if smap[index] >= max_value and freq[index] < 7:
                                max_value = smap[index]
                                max_index = index

                        for index in range(smap.shape[-1]):
                            if smap[index] >= second_max_value and freq[index] < 7 and index != max_index:
                                second_max_value = smap[index]
                                second_max_index = index

                        x.data[0, max_index] += 0.1
                        x.data[0, second_max_index] += 0.1
                        freq[max_index] += 1
                        freq[second_max_index] += 1

                        cur_iter += 1

            n[j] = torch.norm(x - img) / x.shape[-1]
            j += 1
            pbar.update(1)
        return torch.mean(n)
