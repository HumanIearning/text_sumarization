import torch

print(torch.__version__)
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
# from torch.utils.data import DataLoader
# from ignite.engine import DeterministicEngine, Events
# from ignite.utils import manual_seed
#
#
# def random_train_data_loader(size):
#     data = torch.arange(0, size)
#     return DataLoader(data, batch_size=4, shuffle=True)
#
#
# def print_train_data(engine, batch):
#     i = engine.state.iteration
#     e = engine.state.epoch
#     print("train", e, i, batch.tolist())
#
# trainer = DeterministicEngine(print_train_data)
#
# print("Original Run")
# manual_seed(56)
# trainer.run(random_train_data_loader(38), max_epochs=2, epoch_length=5)
#
# print("Resumed Run")
# # Resume from 2nd epoch
# trainer.load_state_dict({"epoch": 1, "epoch_length": 6, "max_epochs": 2, "rng_states": None})
# manual_seed(56)
# trainer.run(random_train_data_loader(38))
#
# data = torch.arange(0, 39)
# a = DataLoader(data, batch_size=4, shuffle=True)
# for i in a:
#     print(i)
#





#
# import matplotlib.pyplot as plt
# import numpy as np
#
# w, b = np.random.randint(1,10), np.random.randint(-10,10)
# x = np.random.normal(0, 1, size=(100, ))
# y = w*x + b
# noise = np.random.normal(0, 0.3, (100,))
# y = y + noise
# y = 1/(1 + np.exp(-y))
#
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.scatter(x, y, alpha=0.5)
# ax.tick_params(labelsize=10)
#
# plt.show()



# import torch
# max_length = 2048
# d_model = 768
#
# pos = torch.arange(0, max_length).unsqueeze(-1).float()
# dim = torch.arange(0, d_model // 2).float()
#
# pos_enc = torch.FloatTensor(d_model, max_length).zero_()
#
# print(pos, pos.shape)
#
# sin = torch.sin(pos / 1e+4 ** (dim / float(d_model)))
# # print(dim, dim.shape)
# print(pos / 10e+4 ** (dim / float(d_model)))
# # print(sin, sin.shape)

# print(pos_enc[:, 0::2].shape)
