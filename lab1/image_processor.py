"""
Image Processing Module for Mosaic Generation
============================================

This module handles image preprocessing, grid division, and vectorized operations
for the mosaic generator.

Author: CS5130 Student
Date: 2024
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image

class ImageProcessor:
    """
    Handles image preprocessing and grid operations for mosaic generation.
    """
    
    def __init__(self, grid_size: int = 32):
        """
        Initialize image processor.
        
        Args:
            grid_size (int): Size of the grid (grid_size x grid_size)
        """
        self.grid_size = grid_size
    
    def preprocess_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess input image for mosaic generation.
        
        Args:
            image (np.ndarray): Input image
            target_size (Optional[Tuple[int, int]]): Target size (width, height)
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # Already RGB or BGR
                # Assume it's RGB from PIL, no conversion needed
                pass
        
        # Apply optional color quantization for simplified colors
        image = self.apply_color_quantization(image, levels=16)
        
        # Calculate target size that's divisible by grid_size
        if target_size is None:
            h, w = image.shape[:2]
            # Make dimensions divisible by grid_size
            new_h = (h // self.grid_size) * self.grid_size
            new_w = (w // self.grid_size) * self.grid_size
            # Ensure minimum size
            new_h = max(new_h, self.grid_size * 8)  # At least 8x8 grid
            new_w = max(new_w, self.grid_size * 8)
            target_size = (new_w, new_h)
        
        # Resize image
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        return image.astype(np.uint8)
    
    def apply_color_quantization(self, image: np.ndarray, levels: int = 16) -> np.ndarray:
        """
        Apply color quantization to simplify color variations.
        
        Args:
            image (np.ndarray): Input image
            levels (int): Number of quantization levels per channel
            
        Returns:
            np.ndarray: Quantized image
        """
        # Quantize each channel
        quantized = image.copy()
        step = 256 // levels
        
        for channel in range(image.shape[2]):
            quantized[:, :, channel] = (image[:, :, channel] // step) * step
        
        return quantized
    
    def divide_into_grid_vectorized(self, image: np.ndarray) -> np.ndarray:
        """
        Divide image into grid using vectorized operations.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Grid cells reshaped as (grid_size^2, cell_h, cell_w, 3)
        """
        h, w = image.shape[:2]
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        
        # Crop to exact grid dimensions
        cropped = image[:self.grid_size * cell_h, :self.grid_size * cell_w]
        
        # Reshape into grid cells using vectorized operations
        # Shape: (grid_size, cell_h, grid_size, cell_w, 3)
        reshaped = cropped.reshape(
            self.grid_size, cell_h, self.grid_size, cell_w, 3
        )
        
        # Transpose to group cells together: (grid_size, grid_size, cell_h, cell_w, 3)
        transposed = reshaped.transpose(0, 2, 1, 3, 4)
        
        # Final reshape to list of cells: (grid_size^2, cell_h, cell_w, 3)
        grid_cells = transposed.reshape(-1, cell_h, cell_w, 3)
        
        return grid_cells
    
    def compute_average_colors_vectorized(self, grid_cells: np.ndarray) -> np.ndarray:
        """
        Compute average colors for all grid cells using vectorized operations.
        
        Args:
            grid_cells (np.ndarray): Grid cells (N, cell_h, cell_w, 3)
            
        Returns:
            np.ndarray: Average RGB colors (N, 3)
        """
        # Vectorized average computation across spatial dimensions
        avg_colors = np.mean(grid_cells, axis=(1, 2))
        return avg_colors.astype(np.uint8)
    
    def convert_rgb_to_lab_vectorized(self, rgb_colors: np.ndarray) -> np.ndarray:
        """
        Convert RGB colors to LAB color space using vectorized operations.
        
        Args:
            rgb_colors (np.ndarray): RGB colors (N, 3) or (H, W, 3)
            
        Returns:
            np.ndarray: LAB colors in same shape
        """
        original_shape = rgb_colors.shape
        
        # Ensure uint8 format
        rgb_colors = np.clip(rgb_colors, 0, 255).astype(np.uint8)
        
        if len(original_shape) == 2:  # Multiple colors (N, 3)
            # Create a temporary image for batch conversion
            temp_image = rgb_colors.reshape(1, -1, 3)
        else:  # Image format (H, W, 3)
            temp_image = rgb_colors
        
        try:
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR)
            # Convert BGR to LAB
            lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
            
            # Reshape back to original shape
            lab_colors = lab_image.reshape(original_shape)
            return lab_colors.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: LAB conversion failed, using RGB: {e}")
            return rgb_colors.astype(np.float32)
    
    def reconstruct_mosaic_vectorized(self, tile_indices: np.ndarray, tiles: list, 
                                    target_size: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct mosaic image from tile indices using vectorized operations.
        
        Args:
            tile_indices (np.ndarray): Indices of selected tiles (grid_size^2,)
            tiles (list): List of tile images
            target_size (Tuple[int, int]): Target size (width, height)
            
        Returns:
            np.ndarray: Reconstructed mosaic image
        """
        h, w = target_size[1], target_size[0]
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        
        # Pre-allocate mosaic array
        mosaic = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Resize all selected tiles at once
        selected_tiles = []
        for idx in tile_indices:
            if idx < len(tiles):
                tile = tiles[idx]
                resized_tile = cv2.resize(tile, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                selected_tiles.append(resized_tile)
            else:
                # Fallback gray tile
                gray_tile = np.full((cell_h, cell_w, 3), 128, dtype=np.uint8)
                selected_tiles.append(gray_tile)
        
        selected_tiles = np.array(selected_tiles)  # Shape: (grid_size^2, cell_h, cell_w, 3)
        
        # Reshape tiles back to grid format
        grid_tiles = selected_tiles.reshape(self.grid_size, self.grid_size, cell_h, cell_w, 3)
        
        # Reconstruct mosaic using vectorized operations
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                start_h = i * cell_h
                end_h = start_h + cell_h
                start_w = j * cell_w
                end_w = start_w + cell_w
                
                mosaic[start_h:end_h, start_w:end_w] = grid_tiles[i, j]
        
        return mosaic
    
    def create_comparison_image(self, original: np.ndarray, mosaic: np.ndarray) -> np.ndarray:
        """
        Create a side-by-side comparison image.
        
        Args:
            original (np.ndarray): Original image
            mosaic (np.ndarray): Mosaic image
            
        Returns:
            np.ndarray: Comparison image
        """
        # Ensure same height
        h = min(original.shape[0], mosaic.shape[0])
        original_resized = cv2.resize(original, (original.shape[1], h))
        mosaic_resized = cv2.resize(mosaic, (mosaic.shape[1], h))
        
        # Concatenate horizontally
        comparison = np.hstack([original_resized, mosaic_resized])
        return comparison
