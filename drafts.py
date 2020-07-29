import torch

# a = torch.randint(0,10,(2,3))
# print(a)
pos = (1,2)
def send(pos):
    print(pos)
for i in range(5):
    # print('Hana in my heart')
    send(pos)