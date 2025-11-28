from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
from typing import List, Optional, Dict, Any

class ProgressMonitor(Callback):
    def __init__(self, epochs: int, metrics: List[str] = None) -> None:
        super().__init__()
        self.epochs = epochs
        self.metrics = metrics or ['loss', 'val_loss']
        self.pbar = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        print("Initializing training...")
        self.pbar = tqdm(total=self.epochs, desc="Training")

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        print(f"\nStarting epoch {epoch+1}/{self.epochs}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs is None:
            logs = {}
        
        self.pbar.update(1)
        
        # Format metric values
        values = [f"{logs.get(metric, 0):.4f}" for metric in self.metrics]
        metrics_str = " - ".join(f"{m}: {v}" for m, v in zip(self.metrics, values))
        self.pbar.set_postfix_str(metrics_str)
        
        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{self.epochs} completed. "
            f"Loss: {logs.get('loss', 0):.4f}, "
            f"Val Loss: {logs.get('val_loss', 0):.4f}"
        )

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.pbar.close()
        print("Training completed.")