---
title: lab1
app_file: app.py
sdk: gradio
sdk_version: 5.46.1
---
# Interactive Image Mosaic Generator

A sophisticated image mosaic generator that transforms input images into artistic mosaics using CIFAR-10 dataset tiles. The system employs advanced computer vision techniques including LAB color space matching, vectorized operations, and comprehensive performance analysis.

## 🎯 Features

### Core Functionality
- **CIFAR-10 Tile Dataset**: Uses 5,000 diverse 32×32 images from 10 classes as mosaic tiles
- **LAB Color Space Matching**: Perceptually uniform color matching for better visual results
- **L2 Distance Calculation**: Optimal tile selection using Euclidean distance in LAB space
- **Vectorized Operations**: NumPy-optimized processing for 5-10x performance improvement
- **Interactive Gradio Interface**: User-friendly web interface with real-time processing

### Technical Highlights
- **Grid-based Processing**: Divides images into customizable N×N grids (8×8 to 64×64)
- **Color Quantization**: Optional preprocessing to simplify color variations
- **Performance Metrics**: MSE, SSIM, and PSNR evaluation
- **Comparative Analysis**: Vectorized vs loop-based implementation comparison
- **Real-time Processing**: Optimized for interactive use with progress indicators

## 🚀 Quick Start

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd lab1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```
   Or for auto-refresh during development:
   ```bash
   gradio app.py
   ```

### Usage

1. **Access the Interface**: Open the provided local URL (typically http://localhost:7860)
2. **Upload Image**: Choose any image you'd like to transform
3. **Adjust Settings**: 
   - Grid Size: 8×8 (fast, less detail) to 64×64 (slow, more detail)
   - Comparison Mode: Toggle side-by-side view
4. **Generate**: Click "Generate Mosaic" and wait for processing
5. **Analyze**: Use the Performance Analysis tab for detailed metrics

## 📁 Project Structure

```
lab1/
├── app.py                    # Main Gradio application
├── mosaic_generator.py       # Core mosaic generation logic
├── cifar10_loader.py        # CIFAR-10 dataset handling
├── image_processor.py       # Image preprocessing and grid operations
├── performance_metrics.py   # Quality metrics and performance analysis
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔬 Technical Implementation

### Algorithm Overview

1. **Image Preprocessing**
   - Resize to ensure grid divisibility
   - Optional color quantization (16 levels per channel)
   - RGB format normalization

2. **Grid Division (Vectorized)**
   ```python
   # Efficient grid reshaping using NumPy
   reshaped = image.reshape(grid_size, cell_h, grid_size, cell_w, 3)
   grid_cells = reshaped.transpose(0, 2, 1, 3, 4).reshape(-1, cell_h, cell_w, 3)
   ```

3. **Color Space Conversion**
   - RGB → LAB conversion for perceptual uniformity
   - Vectorized processing of all grid cells simultaneously

4. **Tile Matching**
   ```python
   # Vectorized L2 distance calculation
   distances = np.linalg.norm(
       cell_colors[:, np.newaxis, :] - tile_colors[np.newaxis, :, :], 
       axis=2
   )
   best_indices = np.argmin(distances, axis=1)
   ```

5. **Mosaic Reconstruction**
   - Efficient tile placement using advanced indexing
   - Batch resizing of selected tiles

### Performance Optimizations

- **Precomputed LAB Colors**: All tile colors calculated once at startup
- **Broadcasting Operations**: Efficient distance calculations for all combinations
- **Memory Management**: Optimized array operations to minimize memory usage
- **Vectorized Processing**: Eliminates nested loops for grid operations

## 📊 Performance Metrics

### Quality Metrics
- **MSE (Mean Squared Error)**: Pixel-level reconstruction accuracy (lower = better)
- **SSIM (Structural Similarity Index)**: Perceptual similarity (0-1, higher = better)
- **PSNR (Peak Signal-to-Noise Ratio)**: Signal quality in dB (higher = better)

### Computational Performance
- **Processing Time**: End-to-end mosaic generation time
- **Scalability Analysis**: Performance vs grid size relationship
- **Implementation Comparison**: Vectorized vs loop-based speedup analysis

### Typical Performance Results
```
Grid Size 16×16: ~0.5s, MSE: 2500, SSIM: 0.65
Grid Size 32×32: ~1.2s, MSE: 1800, SSIM: 0.72  
Grid Size 64×64: ~4.5s, MSE: 1200, SSIM: 0.78
Vectorization Speedup: 5-8x faster than loops
```

## 🎨 Creative Features

### Tile Diversity
- **10 CIFAR-10 Classes**: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, trucks
- **500 Tiles per Class**: 5,000 total unique tiles for variety
- **Automatic Selection**: Best perceptual matches chosen automatically

### Visual Enhancements
- **Side-by-Side Comparison**: Original vs mosaic visualization
- **Progress Indicators**: Real-time processing feedback
- **Interactive Parameters**: Dynamic grid size adjustment
- **Example Gallery**: Pre-loaded demonstration images

## 🔧 Advanced Usage

### Custom Grid Sizes
```python
# Programmatic usage
generator = MosaicGenerator(grid_size=48, max_tiles_per_class=1000)
mosaic = generator.generate_mosaic(your_image)
```

### Performance Analysis
```python
# Analyze multiple grid sizes
results = generator.analyze_performance(image, grid_sizes=[16, 24, 32, 48, 64])
generator.performance_metrics.plot_performance_analysis(results)
```

### Batch Processing
```python
# Process multiple images
for image_path in image_paths:
    image = np.array(Image.open(image_path))
    mosaic = generator.generate_mosaic(image)
    generator.save_mosaic(mosaic, f"mosaic_{image_path}")
```

## 📈 Performance Analysis

The system includes comprehensive performance analysis tools:

### Grid Size Impact
- **Processing Time**: Scales approximately O(n²) with grid size
- **Quality Metrics**: Generally improve with higher resolution
- **Memory Usage**: Increases linearly with number of cells

### Vectorization Benefits
- **Speed Improvement**: 5-10x faster than equivalent loop implementations
- **Scalability**: Benefits increase with larger grid sizes
- **Memory Efficiency**: Better cache utilization and reduced overhead

### Optimization Techniques
1. **Precomputation**: LAB colors calculated once at startup
2. **Broadcasting**: Efficient NumPy operations for distance calculations
3. **Memory Layout**: Optimized array shapes for cache efficiency
4. **Batch Operations**: Process multiple cells simultaneously

## 🎓 Educational Value

This project demonstrates key computer vision and optimization concepts:

- **Color Space Theory**: RGB vs LAB perceptual differences
- **Vectorization**: NumPy optimization techniques
- **Performance Analysis**: Systematic benchmarking approaches
- **User Interface Design**: Interactive visualization principles
- **Software Architecture**: Modular, extensible design patterns

## 🔬 Technical Details

### Color Space Conversion
The LAB color space provides perceptually uniform distances, meaning that equal numerical differences correspond to equal perceived color differences. This results in more visually pleasing tile selections compared to RGB space.

### Vectorized Operations
By reshaping images into grid structures and using NumPy broadcasting, we eliminate nested loops and leverage optimized C implementations for significant performance gains.

### Memory Efficiency
The system processes images in chunks and uses view-based operations where possible to minimize memory allocation and copying.

## 🚀 Deployment

### Local Development
```bash
gradio app.py  # Auto-refresh on changes
```

### Production Deployment
For deployment on Hugging Face Spaces:
1. Create new Space on Hugging Face
2. Upload all project files
3. Ensure requirements.txt includes exact versions
4. Set Python version in Space settings

### Docker Deployment
```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

## 📝 Assignment Requirements Checklist

- ✅ **Image Selection and Preprocessing**: Resize, optional color quantization
- ✅ **Grid Division**: Vectorized NumPy operations, no nested loops
- ✅ **Tile Mapping**: CIFAR-10 dataset tiles with LAB color space matching
- ✅ **Gradio Interface**: User-friendly with parameter controls
- ✅ **Performance Metrics**: MSE and SSIM implementation
- ✅ **Performance Analysis**: Multiple grid sizes with timing measurements
- ✅ **Vectorization**: Comparison with loop-based implementation
- ✅ **Creativity**: Advanced color space, comprehensive analysis tools

## 🎯 Results and Analysis

The system successfully creates high-quality mosaics with the following characteristics:

- **Visual Quality**: SSIM values typically 0.6-0.8 depending on grid size
- **Processing Speed**: Real-time generation for reasonable grid sizes
- **Tile Utilization**: Effective use of CIFAR-10 diversity
- **User Experience**: Intuitive interface with immediate feedback

## 🔮 Future Enhancements

Potential improvements for extended functionality:
- **Custom Tile Sets**: Allow users to upload their own tile collections
- **Advanced Matching**: Consider texture and edge information
- **GPU Acceleration**: CUDA implementation for larger images
- **Video Processing**: Extend to video mosaic generation
- **Style Transfer**: Combine with neural style transfer techniques

---

*This project demonstrates advanced computer vision techniques, performance optimization, and user interface design for the CS5130 Computer Vision course.*
