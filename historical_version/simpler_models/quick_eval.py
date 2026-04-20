import torch

model = torch.load("model_save.pt", weights_only=False)
model.eval()

tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float32, device = "mps")/10 + 0.5
print(tensor)
print(model(tensor))