"""
Flower Federated Learning Client with GPU Support and TLS
Each client trains a local model on its own data partition.
"""

import argparse
import warnings
from collections import OrderedDict
from typing import Tuple
from pathlib import Path
import logging

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from prometheus_client import start_http_server, Counter, Gauge, Histogram

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
TRAINING_ROUNDS = Counter(
    "client_training_rounds_total",
    "Total training rounds completed by this client",
    labelnames=("client_id", "round"),
)
TRAINING_DURATION = Histogram(
    "client_training_duration_seconds",
    "Duration of local training in seconds",
    labelnames=("client_id", "round"),
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
LOCAL_ACCURACY = Gauge(
    "client_local_accuracy",
    "Local model accuracy",
    labelnames=("client_id", "round"),
)
LOCAL_LOSS = Gauge(
    "client_local_loss",
    "Local model loss",
    labelnames=("client_id", "round"),
)
SAMPLES_TRAINED = Counter(
    "client_samples_trained_total",
    "Total samples trained locally (cumulative)",
    labelnames=("client_id",),
)

class Net(nn.Module):
    """CNN for image classification with batch normalization."""
    
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_data(client_id: int, num_clients: int) -> Tuple[DataLoader, DataLoader]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = datasets.CIFAR10(root="/app/data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root="/app/data", train=False, download=True, transform=transform_test)
    
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] += len(trainset) - sum(lengths)
    
    datasets_split = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))
    trainset_client = datasets_split[client_id]
    
    trainloader = DataLoader(trainset_client, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=32, num_workers=2)
    
    return trainloader, testloader


def train(net, trainloader, epochs: int, device: str, learning_rate: float = 0.001):
    import time
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total
        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

    duration = time.time() - start_time
    final_loss = epoch_loss
    final_acc = epoch_acc
    return final_loss, final_acc, duration


def test(net, testloader, device: str, client_id: str = "unknown", server_round: str = "0") -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    loss = loss / len(testloader)
    
    # Update Prometheus metrics with labels
    LOCAL_ACCURACY.labels(client_id=client_id, round=server_round).set(accuracy)
    LOCAL_LOSS.labels(client_id=client_id, round=server_round).set(loss)
    
    return loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    """Flower client implementing PyTorch-based federated learning."""
    
    def __init__(self, net, trainloader, testloader, device):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.client_id = "unknown"
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def set_parameters(self, parameters):
        state_dict = OrderedDict(zip(self.net.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.net.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        epochs = config.get("local_epochs", 1)
        learning_rate = config.get("learning_rate", 0.001)
        server_round = str(config.get("server_round", 0))

        logger.info(f"Round {server_round}: Training for {epochs} epoch(s) with LR={learning_rate}")

        loss, acc, duration = train(self.net, self.trainloader, epochs, self.device, learning_rate)

        # update Prometheus metrics
        LOCAL_ACCURACY.labels(client_id=self.client_id, round=server_round).set(acc)
        LOCAL_LOSS.labels(client_id=self.client_id, round=server_round).set(loss)
        TRAINING_DURATION.labels(client_id=self.client_id, round=server_round).observe(duration)
        TRAINING_ROUNDS.labels(client_id=self.client_id, round=server_round).inc()
        SAMPLES_TRAINED.labels(client_id=self.client_id).inc(len(self.trainloader.dataset))

        metrics = {
            "accuracy": acc,
            "loss": loss,
            "training_time": duration,
            "samples": len(self.trainloader.dataset),
        }

        return self.get_parameters(config={}), len(self.trainloader.dataset), metrics

    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        server_round = str(config.get("server_round", 0))
        loss, accuracy = test(self.net, self.testloader, self.device, client_id=self.client_id, server_round=server_round)
        logger.info(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.testloader.dataset), {"accuracy": accuracy, "loss": loss}


def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--server_address", type=str, default="localhost:8080")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--metrics_port", type=int, default=9092)
    parser.add_argument("--enable_tls", action="store_true")
    parser.add_argument("--ca_cert_path", type=str, default="/app/certs/ca.crt")
    args = parser.parse_args()
    
    # Start Prometheus metrics server
    try:
        metrics_port = args.metrics_port + args.client_id
        start_http_server(metrics_port)
        logger.info(f"Prometheus metrics server started on port {metrics_port}")
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Client {args.client_id} using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")

    trainloader, testloader = load_data(args.client_id, args.num_clients)
    logger.info(f"Training samples: {len(trainloader.dataset)}, Test samples: {len(testloader.dataset)}")
    
    net = Net(num_classes=10).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

    flower_client = FlowerClient(net, trainloader, testloader, device)
    flower_client.client_id = str(args.client_id)

    if args.enable_tls:
        ca_cert_path = Path(args.ca_cert_path)
        if not ca_cert_path.exists():
            logger.error(f"CA certificate not found: {ca_cert_path}")
            raise FileNotFoundError(f"CA certificate not found: {ca_cert_path}")
        root_certificates = ca_cert_path.read_bytes()
        logger.info(f"TLS enabled - Connecting to {args.server_address}")
        fl.client.start_client(
            server_address=args.server_address,
            client=flower_client.to_client(),
            root_certificates=root_certificates,
        )
    else:
        logger.info(f"Connecting to server at {args.server_address} (TLS disabled)")
        fl.client.start_client(
            server_address=args.server_address,
            client=flower_client.to_client(),
        )


if __name__ == "__main__":
    main()
