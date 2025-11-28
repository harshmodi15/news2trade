import tensorflow as tf
import logging
from typing import List, Optional

class GPUManager:
    """Utility class for managing GPU configurations and setup."""
    
    @staticmethod
    def setup_gpu() -> None:
        """
        Configure GPU settings and print device information.
        Should be called at the start of the program.
        """
        logging.info(f"TensorFlow version: {tf.__version__}")
        
        # Get available GPUs
        gpus: List = tf.config.list_physical_devices('GPU')
        
        if gpus:
            logging.info(f"Number of GPUs Available: {len(gpus)}")
            logging.info(f"GPU Devices: {[gpu.name for gpu in gpus]}")
            
            # Configure memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info("GPU memory growth enabled")
            except RuntimeError as e:
                logging.error(f"Error configuring GPU memory growth: {str(e)}")
        else:
            logging.warning("No GPU devices available. Running on CPU.")
    
    @staticmethod
    def get_gpu_info() -> dict:
        """
        Get detailed information about available GPU devices.
        
        Returns:
            dict: Dictionary containing GPU information
        """
        return {
            'version': tf.__version__,
            'gpu_available': bool(tf.config.list_physical_devices('GPU')),
            'gpu_name': tf.test.gpu_device_name(),
            'num_gpus': len(tf.config.list_physical_devices('GPU')),
            'devices': str(tf.config.list_physical_devices())
        }

    @staticmethod
    def set_gpu_memory_limit(memory_limit: int) -> None:
        """
        Set memory limit for GPU devices.
        
        Args:
            memory_limit: Memory limit in MB
        """
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit
                        )]
                    )
                logging.info(f"GPU memory limit set to {memory_limit}MB")
            except RuntimeError as e:
                logging.error(f"Error setting GPU memory limit: {str(e)}")