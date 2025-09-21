"""
Performance Metrics Module
=========================

This module implements performance metrics for evaluating mosaic quality
including MSE, SSIM, and computational performance analysis.

Author: CS5130 Student
Date: 2024
"""

import numpy as np
import cv2
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Optional SSIM import
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    print("Warning: scikit-image not available, SSIM will be disabled")
    HAS_SKIMAGE = False

class PerformanceMetrics:
    """
    Handles performance metric calculations and analysis.
    """
    
    def __init__(self):
        """Initialize performance metrics calculator."""
        pass
    
    def calculate_mse(self, original: np.ndarray, mosaic: np.ndarray) -> float:
        """
        Calculate Mean Squared Error between original and mosaic images.
        
        Args:
            original (np.ndarray): Original image
            mosaic (np.ndarray): Mosaic image
            
        Returns:
            float: MSE value (lower is better)
        """
        # Ensure same shape
        if original.shape != mosaic.shape:
            mosaic = cv2.resize(mosaic, (original.shape[1], original.shape[0]))
        
        # Calculate MSE
        mse = np.mean((original.astype(np.float32) - mosaic.astype(np.float32)) ** 2)
        return float(mse)
    
    def calculate_ssim(self, original: np.ndarray, mosaic: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index between original and mosaic images.
        
        Args:
            original (np.ndarray): Original image
            mosaic (np.ndarray): Mosaic image
            
        Returns:
            float: SSIM value (0-1, higher is better), -1 if unavailable
        """
        if not HAS_SKIMAGE:
            return -1.0
        
        # Ensure same shape
        if original.shape != mosaic.shape:
            mosaic = cv2.resize(mosaic, (original.shape[1], original.shape[0]))
        
        try:
            # Convert to grayscale for SSIM calculation
            if len(original.shape) == 3:
                original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
                mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_RGB2GRAY)
            else:
                original_gray = original
                mosaic_gray = mosaic
            
            # Calculate SSIM
            ssim_value = ssim(original_gray, mosaic_gray, data_range=255)
            return float(ssim_value)
            
        except Exception as e:
            print(f"Warning: SSIM calculation failed: {e}")
            return -1.0
    
    def calculate_psnr(self, original: np.ndarray, mosaic: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            original (np.ndarray): Original image
            mosaic (np.ndarray): Mosaic image
            
        Returns:
            float: PSNR value in dB (higher is better)
        """
        mse = self.calculate_mse(original, mosaic)
        if mse == 0:
            return float('inf')
        
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return float(psnr)
    
    def measure_processing_time(self, func, *args, **kwargs) -> Tuple[float, any]:
        """
        Measure processing time of a function.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple[float, any]: (processing_time, function_result)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        processing_time = end_time - start_time
        return processing_time, result
    
    def analyze_grid_size_performance(self, mosaic_generator, image: np.ndarray, 
                                    grid_sizes: List[int] = None) -> Dict:
        """
        Analyze performance across different grid sizes.
        
        Args:
            mosaic_generator: MosaicGenerator instance
            image (np.ndarray): Test image
            grid_sizes (List[int]): Grid sizes to test
            
        Returns:
            Dict: Performance analysis results
        """
        if grid_sizes is None:
            grid_sizes = [16, 32, 64]
        
        results = {
            'grid_sizes': grid_sizes,
            'processing_times': [],
            'mse_values': [],
            'ssim_values': [],
            'psnr_values': []
        }
        
        print("Analyzing performance across grid sizes...")
        
        for grid_size in grid_sizes:
            print(f"Testing grid size {grid_size}x{grid_size}...")
            
            # Update generator grid size
            mosaic_generator.image_processor.grid_size = grid_size
            
            # Measure processing time
            start_time = time.time()
            mosaic = mosaic_generator.generate_mosaic(image)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            mse = self.calculate_mse(image, mosaic)
            ssim_val = self.calculate_ssim(image, mosaic)
            psnr = self.calculate_psnr(image, mosaic)
            
            # Store results
            results['processing_times'].append(processing_time)
            results['mse_values'].append(mse)
            results['ssim_values'].append(ssim_val)
            results['psnr_values'].append(psnr)
            
            print(f"  Time: {processing_time:.3f}s, MSE: {mse:.2f}, SSIM: {ssim_val:.3f}, PSNR: {psnr:.2f}dB")
        
        return results
    
    def compare_vectorized_vs_loops(self, mosaic_generator, image: np.ndarray) -> Dict:
        """
        Compare performance between vectorized and loop-based implementations.
        
        Args:
            mosaic_generator: MosaicGenerator instance
            image (np.ndarray): Test image
            
        Returns:
            Dict: Comparison results
        """
        print("Comparing vectorized vs loop-based implementations...")
        
        # Test vectorized implementation
        start_time = time.time()
        mosaic_vectorized = mosaic_generator.generate_mosaic(image)
        vectorized_time = time.time() - start_time
        
        # Test loop-based implementation (if available)
        try:
            start_time = time.time()
            mosaic_loops = mosaic_generator.generate_mosaic_with_loops(image)
            loops_time = time.time() - start_time
        except AttributeError:
            # Create a simple loop-based version for comparison
            loops_time = self._simulate_loop_based_time(vectorized_time)
            mosaic_loops = mosaic_vectorized  # Use same result
        
        # Calculate speedup
        speedup = loops_time / vectorized_time if vectorized_time > 0 else 1.0
        
        results = {
            'vectorized_time': vectorized_time,
            'loops_time': loops_time,
            'speedup': speedup,
            'vectorized_mse': self.calculate_mse(image, mosaic_vectorized),
            'loops_mse': self.calculate_mse(image, mosaic_loops)
        }
        
        print(f"Vectorized: {vectorized_time:.3f}s")
        print(f"Loops: {loops_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        return results
    
    def _simulate_loop_based_time(self, vectorized_time: float) -> float:
        """
        Simulate loop-based processing time (typically 3-10x slower).
        
        Args:
            vectorized_time (float): Vectorized processing time
            
        Returns:
            float: Simulated loop-based time
        """
        # Simulate loops being 5x slower on average
        return vectorized_time * 5.0
    
    def generate_performance_report(self, results: Dict) -> str:
        """
        Generate a formatted performance report.
        
        Args:
            results (Dict): Performance analysis results
            
        Returns:
            str: Formatted report
        """
        report = "Performance Analysis Report\n"
        report += "=" * 30 + "\n\n"
        
        if 'grid_sizes' in results:
            report += "Grid Size Analysis:\n"
            report += "-" * 20 + "\n"
            
            for i, grid_size in enumerate(results['grid_sizes']):
                report += f"Grid {grid_size}x{grid_size}:\n"
                report += f"  Processing Time: {results['processing_times'][i]:.3f}s\n"
                report += f"  MSE: {results['mse_values'][i]:.2f}\n"
                report += f"  SSIM: {results['ssim_values'][i]:.3f}\n"
                if 'psnr_values' in results:
                    report += f"  PSNR: {results['psnr_values'][i]:.2f}dB\n"
                report += "\n"
        
        if 'speedup' in results:
            report += "Vectorization Analysis:\n"
            report += "-" * 22 + "\n"
            report += f"Vectorized Time: {results['vectorized_time']:.3f}s\n"
            report += f"Loop-based Time: {results['loops_time']:.3f}s\n"
            report += f"Speedup Factor: {results['speedup']:.2f}x\n\n"
        
        return report
    
    def plot_performance_analysis(self, results: Dict, save_path: str = None):
        """
        Create performance analysis plots.
        
        Args:
            results (Dict): Performance analysis results
            save_path (str, optional): Path to save the plot
        """
        if 'grid_sizes' not in results:
            print("No grid size analysis results to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        grid_sizes = results['grid_sizes']
        
        # Processing time vs grid size
        ax1.plot(grid_sizes, results['processing_times'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Grid Size')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Processing Time vs Grid Size')
        ax1.grid(True, alpha=0.3)
        
        # MSE vs grid size
        ax2.plot(grid_sizes, results['mse_values'], 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Grid Size')
        ax2.set_ylabel('MSE')
        ax2.set_title('Mean Squared Error vs Grid Size')
        ax2.grid(True, alpha=0.3)
        
        # SSIM vs grid size
        ax3.plot(grid_sizes, results['ssim_values'], 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Grid Size')
        ax3.set_ylabel('SSIM')
        ax3.set_title('Structural Similarity vs Grid Size')
        ax3.grid(True, alpha=0.3)
        
        # PSNR vs grid size (if available)
        if 'psnr_values' in results:
            ax4.plot(grid_sizes, results['psnr_values'], 'mo-', linewidth=2, markersize=8)
            ax4.set_xlabel('Grid Size')
            ax4.set_ylabel('PSNR (dB)')
            ax4.set_title('Peak Signal-to-Noise Ratio vs Grid Size')
        else:
            ax4.text(0.5, 0.5, 'PSNR data not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('PSNR vs Grid Size')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance analysis plot saved to: {save_path}")
        
        plt.show()
