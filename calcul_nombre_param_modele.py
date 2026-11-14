import torch

checkpoint = torch.load("model.pth", map_location="cpu")

if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
    state_dict = checkpoint
elif "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    raise KeyError("Impossible de trouver les poids du modèle dans le checkpoint.")

total_params = sum(value.numel() for value in state_dict.values())

print(f"Nombre total de paramètres : {total_params:,}")