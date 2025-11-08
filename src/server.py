"""
Flower Federated Learning Server with TLS and Monitoring
This server orchestrates federated learning with security and observability.
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, Parameters
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import argparse
import pickle
import logging
from datetime import datetime

# Prometheus metrics
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
# ROUNDS_COMPLETED = Counter('fl_rounds_completed', 'Total FL rounds completed')
# CLIENTS_CONNECTED = Gauge('fl_clients_connected', 'Number of connected clients')
# ROUND_DURATION = Histogram('fl_round_duration_seconds', 'Duration of FL rounds')
# AGGREGATION_ACCURACY = Gauge('fl_aggregation_accuracy', 'Global model accuracy')
# AGGREGATION_LOSS = Gauge('fl_aggregation_loss', 'Global model loss')

# counters should use _total suffix
ROUNDS_COMPLETED = Counter(
    "fl_rounds_completed_total",
    "Total FL rounds completed",
    labelnames=("round",),
)

# Number of currently connected/participating clients (labelled by round)
CLIENTS_CONNECTED = Gauge(
    "fl_clients_connected",
    "Number of connected clients",
    labelnames=("round",),
)

# Round duration in seconds (histogram for quantiles)
ROUND_DURATION = Histogram(
    "fl_round_duration_seconds",
    "Duration of FL rounds in seconds",
    labelnames=("round",),
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

AGGREGATION_ACCURACY = Gauge(
    "fl_aggregation_accuracy",
    "Global model accuracy after aggregation",
    labelnames=("round",),
)
AGGREGATION_LOSS = Gauge(
    "fl_aggregation_loss",
    "Global model loss after aggregation",
    labelnames=("round",),
)

# Extra useful metrics
AGGREGATED_EXAMPLES = Gauge(
    "fl_aggregated_examples",
    "Total number of examples aggregated in this round",
    labelnames=("round",),
)
CLIENTS_PARTICIPATING = Gauge(
    "fl_clients_participating",
    "Number of clients who participated in aggregation",
    labelnames=("round",),
)


def weighted_average(metrics: List[Tuple[int, Metrics]], server_round=None) -> Metrics:
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]

    avg_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0
    avg_loss = sum(losses) / total_examples if total_examples > 0 else 0

    label = str(server_round) if server_round is not None else "unknown"

    AGGREGATION_ACCURACY.labels(round=label).set(avg_accuracy)
    AGGREGATION_LOSS.labels(round=label).set(avg_loss)
    AGGREGATED_EXAMPLES.labels(round=label).set(total_examples)

    logger.info(f"Aggregated metrics - Round {label} - Accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}, Examples: {total_examples}")

    return {"accuracy": avg_accuracy, "loss": avg_loss}

def fit_config(server_round: int) -> Dict:
    """
    Configure training for each round.
    
    Args:
        server_round: Current federated learning round
    
    Returns:
        Configuration dictionary sent to clients
    """
    config = {
        "server_round": server_round,
        "local_epochs": 1 if server_round < 3 else 2,  # Adaptive epochs
        "batch_size": 32,
        "learning_rate": 0.001 if server_round < 5 else 0.0005,  # Learning rate decay
    }
    logger.info(f"Round {server_round} config: {config}")
    return config


def evaluate_config(server_round: int) -> Dict:
    """
    Configure evaluation for each round.
    
    Args:
        server_round: Current federated learning round
    
    Returns:
        Configuration dictionary sent to clients
    """
    return {"server_round": server_round}


class CheckpointStrategy(FedAvg):
    """Extended FedAvg strategy with model checkpointing."""
    
    def __init__(self, checkpoint_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def aggregate_fit(self, server_round: int, results, failures):
        start_time = time.time()

        # Update client count (number participating)
        clients_participating = len(results)
        CLIENTS_PARTICIPATING.labels(round=str(server_round)).set(clients_participating)
        CLIENTS_CONNECTED.labels(round=str(server_round)).set(clients_participating)

        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Save checkpoint as before...
        if aggregated_parameters is not None:
            checkpoint_path = self.checkpoint_dir / f"round_{server_round}.pkl"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(aggregated_parameters, f)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # observe round duration and bump rounds counter
        duration = time.time() - start_time
        ROUND_DURATION.labels(round=str(server_round)).observe(duration)
        ROUNDS_COMPLETED.labels(round=str(server_round)).inc()

        # If aggregated_metrics contains accuracy/loss, set them too
        if aggregated_metrics and isinstance(aggregated_metrics, dict):
            if "accuracy" in aggregated_metrics:
                AGGREGATION_ACCURACY.labels(round=str(server_round)).set(aggregated_metrics["accuracy"])
            if "loss" in aggregated_metrics:
                AGGREGATION_LOSS.labels(round=str(server_round)).set(aggregated_metrics["loss"])

        logger.info(f"Round {server_round} completed in {duration:.2f}s with {clients_participating} clients")
        return aggregated_parameters, aggregated_metrics



def load_checkpoint(checkpoint_path: str) -> Optional[Parameters]:
    """Load model checkpoint from disk."""
    path = Path(checkpoint_path)
    if path.exists():
        with open(path, "rb") as f:
            parameters = pickle.load(f)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return parameters
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

# define wrapper so we can include round label inside weighted_average
def evaluate_metrics_agg_fn(metrics: List[Tuple[int, Metrics]], weighted: Optional[List[int]] = None) -> Metrics:
    """
    Aggregate client evaluation metrics with Prometheus logging.
    """
    # Flower passes metrics as list of (num_examples, dict)
    # Use weighted_average to compute weighted accuracy/loss
    return weighted_average(metrics)

def main():
    """Start Flower server with TLS and monitoring."""
    parser = argparse.ArgumentParser(description="Flower Server with TLS and Monitoring")
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (default: 0.0.0.0:8080)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds (default: 10)",
    )
    parser.add_argument(
        "--min_clients",
        type=int,
        default=2,
        help="Minimum number of clients required (default: 2)",
    )
    parser.add_argument(
        "--min_available_clients",
        type=int,
        default=2,
        help="Minimum number of available clients (default: 2)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/app/checkpoints",
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--metrics_port",
        type=int,
        default=9091,
        help="Port for Prometheus metrics (default: 9091)",
    )
    parser.add_argument(
        "--enable_tls",
        action="store_true",
        help="Enable TLS encryption",
    )
    parser.add_argument(
        "--cert_path",
        type=str,
        default="/app/certs/server.crt",
        help="Path to TLS certificate",
    )
    parser.add_argument(
        "--key_path",
        type=str,
        default="/app/certs/server.key",
        help="Path to TLS private key",
    )
    
    args = parser.parse_args()
    
    # Start Prometheus metrics server
    try:
        start_http_server(args.metrics_port)
        logger.info(f"Prometheus metrics server started on port {args.metrics_port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
    
    # Load checkpoint if resuming
    initial_parameters = None
    if args.resume_from:
        initial_parameters = load_checkpoint(args.resume_from)
    
    # Define strategy with checkpointing
    strategy = CheckpointStrategy(
    checkpoint_dir=args.checkpoint_dir,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=args.min_clients,
    min_evaluate_clients=args.min_clients,
    min_available_clients=args.min_available_clients,
    evaluate_metrics_aggregation_fn=evaluate_metrics_agg_fn,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=initial_parameters,
    )
    
    # Configure TLS if enabled
    server_config = fl.server.ServerConfig(num_rounds=args.rounds)
    
    if args.enable_tls:
        cert_path = Path(args.cert_path)
        key_path = Path(args.key_path)
        
        if not cert_path.exists() or not key_path.exists():
            logger.error("TLS enabled but certificate or key not found!")
            logger.error(f"Certificate: {cert_path} (exists: {cert_path.exists()})")
            logger.error(f"Key: {key_path} (exists: {key_path.exists()})")
            raise FileNotFoundError("TLS certificate or key not found")
        
        logger.info(f"TLS enabled - Certificate: {cert_path}, Key: {key_path}")
        
        # Start server with TLS
        fl.server.start_server(
            server_address=args.server_address,
            config=server_config,
            strategy=strategy,
            certificates=(
                cert_path.read_bytes(),
                key_path.read_bytes(),
            ),
        )
    else:
        logger.info(f"Starting Flower server on {args.server_address} (TLS disabled)")
        logger.info(f"Rounds: {args.rounds}, Min clients: {args.min_clients}")
        logger.info("Waiting for clients to connect...")
        
        # Start server without TLS
        fl.server.start_server(
            server_address=args.server_address,
            config=server_config,
            strategy=strategy,
        )


if __name__ == "__main__":
    main()