"""
Interactive Image Mosaic Generator
=================================

Main mosaic generator class that orchestrates the entire pipeline:
- Loads CIFAR-10 tiles
- Processes input images
- Matches tiles using LAB color space L2 distance
- Reconstructs mosaic images

Author: CS5130 Student
Date: 2024
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import time
from PIL import Image

from cifar10_loader import CIFAR10TileLoader
from image_processor import ImageProcessor
from performance_metrics import PerformanceMetrics

class MosaicGenerator:
    """
    Main class for generating image mosaics using CIFAR-10 dataset tiles.
    """
    
    def __init__(self, grid_size: int = 32, tile_size: int = 32, max_tiles_per_class: int = 500):
        """
        Initialize the mosaic generator.
        
        Args:
            grid_size (int): Size of the grid (grid_size x grid_size)
            tile_size (int): Size of each tile in pixels
            max_tiles_per_class (int): Maximum tiles per CIFAR-10 class
        """
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.max_tiles_per_class = max_tiles_per_class
        
        # Initialize components
        print("Initializing Mosaic Generator...")
        self.tile_loader = CIFAR10TileLoader(max_tiles_per_class=max_tiles_per_class)
        self.image_processor = ImageProcessor(grid_size=grid_size)
        self.performance_metrics = PerformanceMetrics()
        
        # Load tiles and precompute LAB colors
        self.tiles, self.tile_lab_colors = self.tile_loader.load_tiles()
        
        print(f"Mosaic Generator initialized with {len(self.tiles)} tiles")
    
    def generate_mosaic(self, input_image: np.ndarray) -> np.ndarray:
        """
        Generate a mosaic from an input image using vectorized operations.
        
        Args:
            input_image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Generated mosaic image
        """
        # Step 1: Preprocess the input image
        processed_image = self.image_processor.preprocess_image(input_image)
        
        # Step 2: Divide image into grid cells
        grid_cells = self.image_processor.divide_into_grid_vectorized(processed_image)
        
        # Step 3: Compute average colors for each cell
        cell_avg_colors = self.image_processor.compute_average_colors_vectorized(grid_cells)
        
        # Step 4: Convert RGB colors to LAB for perceptual matching
        cell_lab_colors = self.image_processor.convert_rgb_to_lab_vectorized(cell_avg_colors)
        
        # Step 5: Find best matching tiles using L2 distance in LAB space
        best_tile_indices = self.tile_loader.find_best_matching_tiles(cell_lab_colors)
        
        # Step 6: Reconstruct mosaic from selected tiles
        target_size = (processed_image.shape[1], processed_image.shape[0])
        mosaic = self.image_processor.reconstruct_mosaic_vectorized(
            best_tile_indices, self.tiles, target_size
        )
        
        return mosaic
    
    def generate_mosaic_with_loops(self, input_image: np.ndarray) -> np.ndarray:
        """
        Generate mosaic using loop-based implementation for performance comparison.
        
        Args:
            input_image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Generated mosaic image
        """
        # Preprocess image
        processed_image = self.image_processor.preprocess_image(input_image)
        h, w = processed_image.shape[:2]
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        
        # Initialize mosaic
        mosaic = np.zeros_like(processed_image)
        
        print("Generating mosaic with loops (for performance comparison)...")
        
        # Loop through each grid cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Extract cell
                start_h = i * cell_h
                end_h = start_h + cell_h
                start_w = j * cell_w
                end_w = start_w + cell_w
                
                cell = processed_image[start_h:end_h, start_w:end_w]
                
                # Calculate average color
                avg_rgb = np.mean(cell, axis=(0, 1))
                avg_lab = self.image_processor.convert_rgb_to_lab_vectorized(avg_rgb.reshape(1, 3))[0]
                
                # Find best matching tile (inefficient loop-based search)
                best_distance = float('inf')
                best_tile_idx = 0
                
                for tile_idx, tile_lab in enumerate(self.tile_lab_colors):
                    distance = np.linalg.norm(avg_lab - tile_lab)
                    if distance < best_distance:
                        best_distance = distance
                        best_tile_idx = tile_idx
                
                # Place best tile
                best_tile = self.tiles[best_tile_idx]
                resized_tile = cv2.resize(best_tile, (cell_w, cell_h))
                mosaic[start_h:end_h, start_w:end_w] = resized_tile
            
            # Progress indicator
            if (i + 1) % 8 == 0:
                print(f"  Processed {i + 1}/{self.grid_size} rows...")
        
        return mosaic
    
    def analyze_performance(self, input_image: np.ndarray, grid_sizes: List[int] = None) -> dict:
        """
        Analyze performance across different grid sizes.
        
        Args:
            input_image (np.ndarray): Test image
            grid_sizes (List[int], optional): Grid sizes to test
            
        Returns:
            dict: Performance analysis results
        """
        if grid_sizes is None:
            grid_sizes = [16, 32, 64]
        
        return self.performance_metrics.analyze_grid_size_performance(
            self, input_image, grid_sizes
        )
    
    def compare_implementations(self, input_image: np.ndarray) -> dict:
        """
        Compare vectorized vs loop-based implementations.
        
        Args:
            input_image (np.ndarray): Test image
            
        Returns:
            dict: Comparison results
        """
        return self.performance_metrics.compare_vectorized_vs_loops(self, input_image)
    
    def calculate_similarity_metrics(self, original: np.ndarray, mosaic: np.ndarray) -> dict:
        """
        Calculate similarity metrics between original and mosaic images.
        
        Args:
            original (np.ndarray): Original image
            mosaic (np.ndarray): Mosaic image
            
        Returns:
            dict: Similarity metrics
        """
        return {
            'mse': self.performance_metrics.calculate_mse(original, mosaic),
            'ssim': self.performance_metrics.calculate_ssim(original, mosaic),
            'psnr': self.performance_metrics.calculate_psnr(original, mosaic)
        }
    
    def create_detailed_analysis(self, input_image: np.ndarray) -> dict:
        """
        Create a comprehensive analysis of the mosaic generation process.
        
        Args:
            input_image (np.ndarray): Input image
            
        Returns:
            dict: Detailed analysis results
        """
        print("Creating detailed analysis...")
        
        # Generate mosaic with timing
        start_time = time.time()
        mosaic = self.generate_mosaic(input_image)
        generation_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self.calculate_similarity_metrics(input_image, mosaic)
        
        # Performance analysis
        performance_results = self.analyze_performance(input_image)
        
        # Implementation comparison
        comparison_results = self.compare_implementations(input_image)
        
        # Create comparison image
        comparison_image = self.image_processor.create_comparison_image(input_image, mosaic)
        
        analysis = {
            'generation_time': generation_time,
            'mosaic': mosaic,
            'comparison_image': comparison_image,
            'metrics': metrics,
            'performance_analysis': performance_results,
            'implementation_comparison': comparison_results,
            'grid_size': self.grid_size,
            'total_tiles': len(self.tiles)
        }
        
        print(f"Analysis completed in {generation_time:.3f}s")
        print(f"MSE: {metrics['mse']:.2f}, SSIM: {metrics['ssim']:.3f}, PSNR: {metrics['psnr']:.2f}dB")
        
        return analysis
    
    def update_grid_size(self, new_grid_size: int):
        """
        Update the grid size for the mosaic generator.
        
        Args:
            new_grid_size (int): New grid size
        """
        self.grid_size = new_grid_size
        self.image_processor.grid_size = new_grid_size
        print(f"Updated grid size to {new_grid_size}x{new_grid_size}")
    
    def get_tile_statistics(self) -> dict:
        """
        Get statistics about the loaded tiles.
        
        Returns:
            dict: Tile statistics
        """
        return {
            'total_tiles': len(self.tiles),
            'tiles_per_class': self.max_tiles_per_class,
            'tile_size': f"{self.tile_size}x{self.tile_size}",
            'classes': self.tile_loader.class_names
        }
    
    def save_mosaic(self, mosaic: np.ndarray, filepath: str):
        """
        Save mosaic image to file.
        
        Args:
            mosaic (np.ndarray): Mosaic image
            filepath (str): Output file path
        """
        # Convert to PIL Image and save
        mosaic_pil = Image.fromarray(mosaic)
        mosaic_pil.save(filepath)
        print(f"Mosaic saved to: {filepath}")
    
    def create_demo_images(self) -> List[np.ndarray]:
        """
        Create sample images for demonstration.
        
        Returns:
            List[np.ndarray]: List of demo images
        """
        demo_images = []
        
        # Gradient image
        gradient = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            gradient[:, i] = [i, 128, 255 - i]
        demo_images.append(gradient)
        
        # Checkerboard pattern
        checker = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    checker[i:i+32, j:j+32] = [255, 100, 100]  # Red
                else:
                    checker[i:i+32, j:j+32] = [100, 100, 255]  # Blue
        demo_images.append(checker)
        
        # Circular pattern
        circle = np.zeros((256, 256, 3), dtype=np.uint8)
        center = 128
        for i in range(256):
            for j in range(256):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                intensity = max(0, min(255, int(255 - dist * 2)))
                circle[i, j] = [intensity, intensity // 2, 255 - intensity]
        demo_images.append(circle)
        
        return demo_images
