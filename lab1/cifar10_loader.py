"""
CIFAR-10 Dataset Loader for Mosaic Tiles
========================================

This module handles loading and preprocessing the CIFAR-10 dataset
to be used as tiles for the mosaic generator.

Author: CS5130 Student
Date: 2024
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import os
import pickle
import tarfile
import requests
from PIL import Image

class CIFAR10TileLoader:
    """
    Loads and manages CIFAR-10 dataset images as tiles for mosaic generation.
    """
    
    def __init__(self, cache_dir: str = "cifar10_cache", max_tiles_per_class: int = 500):
        """
        Initialize CIFAR-10 tile loader.
        
        Args:
            cache_dir (str): Directory to cache processed tiles
            max_tiles_per_class (int): Maximum number of tiles per class to use
        """
        self.cache_dir = cache_dir
        self.max_tiles_per_class = max_tiles_per_class
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.tiles = []
        self.tile_lab_colors = []
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_cifar10(self) -> str:
        """
        Download CIFAR-10 dataset if not already present.
        
        Returns:
            str: Path to the extracted dataset
        """
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        extract_path = "cifar-10-batches-py"
        
        if os.path.exists(extract_path):
            print("CIFAR-10 dataset already exists")
            return extract_path
        
        print("Downloading CIFAR-10 dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        
        # Clean up the tar file
        os.remove(filename)
        
        return extract_path

    def load_cifar10_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CIFAR-10 dataset from downloaded files.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (train_images, train_labels)
        """
        print("Loading CIFAR-10 dataset...")
        
        # Download dataset if needed
        dataset_path = self.download_cifar10()
        
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        # Load all batches
        all_images = []
        all_labels = []
        
        # Load training batches
        for i in range(1, 6):
            batch_file = os.path.join(dataset_path, f'data_batch_{i}')
            batch_data = unpickle(batch_file)
            all_images.append(batch_data[b'data'])
            all_labels.extend(batch_data[b'labels'])
        
        # Load test batch
        test_file = os.path.join(dataset_path, 'test_batch')
        test_data = unpickle(test_file)
        all_images.append(test_data[b'data'])
        all_labels.extend(test_data[b'labels'])
        
        # Concatenate all images
        all_images = np.concatenate(all_images, axis=0)
        all_labels = np.array(all_labels)
        
        # Reshape images from flat to 32x32x3
        all_images = all_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        print(f"Loaded {len(all_images)} CIFAR-10 images")
        return all_images, all_labels
    
    def preprocess_tiles(self, images: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        """
        Preprocess CIFAR-10 images to be used as tiles.
        
        Args:
            images (np.ndarray): Raw CIFAR-10 images
            labels (np.ndarray): Image labels
            
        Returns:
            List[np.ndarray]: List of preprocessed tile images
        """
        print("Preprocessing tiles...")
        processed_tiles = []
        
        # Sample tiles from each class
        for class_id in range(10):
            class_mask = labels == class_id
            class_images = images[class_mask]
            
            # Randomly sample tiles from this class
            num_samples = min(self.max_tiles_per_class, len(class_images))
            indices = np.random.choice(len(class_images), num_samples, replace=False)
            sampled_images = class_images[indices]
            
            print(f"  {self.class_names[class_id]}: {num_samples} tiles")
            
            for img in sampled_images:
                # CIFAR-10 images are already 32x32, just ensure proper format
                processed_img = img.astype(np.uint8)
                processed_tiles.append(processed_img)
        
        print(f"Total processed tiles: {len(processed_tiles)}")
        return processed_tiles
    
    def convert_rgb_to_lab(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to LAB color space.
        
        Args:
            rgb_image (np.ndarray): RGB image
            
        Returns:
            np.ndarray: LAB image
        """
        # OpenCV expects BGR format
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        return lab_image.astype(np.float32)
    
    def compute_average_lab_colors(self, tiles: List[np.ndarray]) -> np.ndarray:
        """
        Compute average LAB colors for all tiles.
        
        Args:
            tiles (List[np.ndarray]): List of tile images
            
        Returns:
            np.ndarray: Array of average LAB colors (N, 3)
        """
        print("Computing average LAB colors for tiles...")
        lab_colors = []
        
        for i, tile in enumerate(tiles):
            # Convert to LAB color space
            lab_tile = self.convert_rgb_to_lab(tile)
            
            # Compute average LAB color
            avg_lab = np.mean(lab_tile, axis=(0, 1))
            lab_colors.append(avg_lab)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(tiles)} tiles...")
        
        return np.array(lab_colors)
    
    def load_tiles(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Load and preprocess all tiles with their LAB colors.
        
        Returns:
            Tuple[List[np.ndarray], np.ndarray]: (tiles, lab_colors)
        """
        # Load CIFAR-10 dataset
        images, labels = self.load_cifar10_dataset()
        
        # Preprocess tiles
        self.tiles = self.preprocess_tiles(images, labels)
        
        # Compute LAB colors
        self.tile_lab_colors = self.compute_average_lab_colors(self.tiles)
        
        print(f"Successfully loaded {len(self.tiles)} tiles with LAB colors")
        return self.tiles, self.tile_lab_colors
    
    def get_tile_by_index(self, index: int) -> np.ndarray:
        """
        Get a specific tile by index.
        
        Args:
            index (int): Tile index
            
        Returns:
            np.ndarray: Tile image
        """
        if 0 <= index < len(self.tiles):
            return self.tiles[index]
        else:
            # Return a default gray tile if index is out of range
            return np.full((32, 32, 3), 128, dtype=np.uint8)
    
    def find_best_matching_tiles(self, target_lab_colors: np.ndarray) -> np.ndarray:
        """
        Find best matching tiles for given LAB colors using L2 distance.
        
        Args:
            target_lab_colors (np.ndarray): Target LAB colors (N, 3)
            
        Returns:
            np.ndarray: Indices of best matching tiles (N,)
        """
        # Vectorized L2 distance calculation
        # target_lab_colors: (N, 3), tile_lab_colors: (M, 3)
        # Result: (N, M) distances
        distances = np.linalg.norm(
            target_lab_colors[:, np.newaxis, :] - self.tile_lab_colors[np.newaxis, :, :],
            axis=2
        )
        
        # Find indices of minimum distances
        best_indices = np.argmin(distances, axis=1)
        return best_indices
