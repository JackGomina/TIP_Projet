"""
pytruc.py

Exemple PyTorch basique : charge MNIST, définit un petit réseau, entraîne
quelques itérations et affiche la perte et l'exactitude.

Usage:
	python pytruc.py

Le script télécharge MNIST automatiquement via torchvision si nécessaire.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.fc = nn.Sequential(
			nn.Linear(28 * 28, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 10),
		)

	def forward(self, x):
		x = self.flatten(x)
		return self.fc(x)


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
	model.train()
	total_loss = 0.0
	correct = 0
	total = 0
	for X, y in dataloader:
		X, y = X.to(device), y.to(device)
		optimizer.zero_grad()
		preds = model(X)
		loss = loss_fn(preds, y)
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * X.size(0)
		_, predicted = preds.max(1)
		correct += (predicted == y).sum().item()
		total += X.size(0)

	avg_loss = total_loss / total
	accuracy = correct / total
	return avg_loss, accuracy


def evaluate(model, dataloader, loss_fn, device):
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			preds = model(X)
			loss = loss_fn(preds, y)
			total_loss += loss.item() * X.size(0)
			_, predicted = preds.max(1)
			correct += (predicted == y).sum().item()
			total += X.size(0)

	avg_loss = total_loss / total
	accuracy = correct / total
	return avg_loss, accuracy


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Device:', device)

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1000)

	model = SimpleNet().to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	epochs = 1
	for epoch in range(1, epochs + 1):
		train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
		test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
		print(f'Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}')

	torch.save(model.state_dict(), 'simple_mnist_model.pth')
	print('Model saved to simple_mnist_model.pth')


if __name__ == '__main__':
	main()
