"""
Interactive Image Mosaic Generator - Gradio Interface
===================================================

This is the main Gradio application that provides a user-friendly interface
for the Image Mosaic Generator using CIFAR-10 dataset tiles.

Author: CS5130 Student
Date: 2024
"""

import gradio as gr
import numpy as np
from PIL import Image
import time
import os
import matplotlib.pyplot as plt

from mosaic_generator import MosaicGenerator
from performance_metrics import PerformanceMetrics

# Global mosaic generator instance - initialize at startup
print("Initializing Mosaic Generator at startup... This may take a moment.")
mosaic_gen = MosaicGenerator(grid_size=32, max_tiles_per_class=500)
print("Mosaic Generator ready!")

def get_generator():
    """Get the pre-initialized mosaic generator."""
    return mosaic_gen

def generate_mosaic_interface(image, grid_size, show_comparison):
    """
    Main interface function for mosaic generation.
    
    Args:
        image: Input image from Gradio
        grid_size (int): Grid size for mosaic
        show_comparison (bool): Whether to show side-by-side comparison
        
    Returns:
        Tuple: (output_image, metrics_text, processing_time_text)
    """
    if image is None:
        return None, "Please upload an image first.", "N/A"
    
    try:
   
        # Start timing
        start_time = time.time()

        # Get pre-initialized generator
        generator = get_generator()
        
        # Update grid size
        generator.update_grid_size(int(grid_size))
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
        else:
            image_array = image
     
        
        # Generate mosaic
        mosaic = generator.generate_mosaic(image_array)
        
        
        # Calculate similarity metrics
        metrics = generator.calculate_similarity_metrics(image_array, mosaic)
        
        # Create output image
        if show_comparison:
            # Side-by-side comparison
            output_image = generator.image_processor.create_comparison_image(image_array, mosaic)
        else:
            # Just the mosaic
            output_image = mosaic

        # Format metrics text
        metrics_text = f"""**Similarity Metrics:**
- MSE: {metrics['mse']:.2f} (lower is better)
- SSIM: {metrics['ssim']:.3f} (higher is better, max=1.0)
- PSNR: {metrics['psnr']:.2f} dB (higher is better)

**Image Info:**
- Grid Size: {grid_size}×{grid_size}
- Total Cells: {int(grid_size)**2}
- Tiles Available: {len(generator.tiles)}"""
        
        # Calculate total processing time (including all operations)
        processing_time = time.time() - start_time
        processing_time_text = f"⏱️ Processing Time: {processing_time:.3f} seconds"
        
        return Image.fromarray(output_image), metrics_text, processing_time_text
        
    except Exception as e:
        error_msg = f"Error generating mosaic: {str(e)}"
        print(error_msg)
        return None, error_msg, "Error"

def analyze_performance_interface(image):
    """
    Interface function for performance analysis.
    
    Args:
        image: Input image from Gradio
        
    Returns:
        Tuple: (analysis_text, plot_image)
    """
    if image is None:
        return "Please upload an image first.", None
    
    try:
        # Get pre-initialized generator
        generator = get_generator()
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
        else:
            image_array = image
        
        print("Running performance analysis...")
        
        # Run performance analysis
        results = generator.analyze_performance(image_array, grid_sizes=[16, 32, 64])
        
        # Generate performance report
        performance_metrics = PerformanceMetrics()
        report = performance_metrics.generate_performance_report(results)
        
        # Create performance plot
        plot_path = "performance_analysis.png"
        performance_metrics.plot_performance_analysis(results, save_path=plot_path)
        
        # Load plot image
        if os.path.exists(plot_path):
            plot_image = Image.open(plot_path)
        else:
            plot_image = None
        
        return report, plot_image
        
    except Exception as e:
        error_msg = f"Error in performance analysis: {str(e)}"
        print(error_msg)
        return error_msg, None

def compare_implementations_interface(image):
    """
    Interface function for comparing vectorized vs loop implementations.
    
    Args:
        image: Input image from Gradio
        
    Returns:
        str: Comparison results
    """
    if image is None:
        return "Please upload an image first."
    
    try:
        # Get pre-initialized generator
        generator = get_generator()
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
        else:
            image_array = image
        
        print("Comparing implementations...")
        
        # Run comparison
        comparison = generator.compare_implementations(image_array)
        
        # Format results
        results_text = f"""**Implementation Comparison:**

🚀 **Vectorized Implementation:**
- Processing Time: {comparison['vectorized_time']:.3f} seconds
- MSE: {comparison['vectorized_mse']:.2f}

🐌 **Loop-based Implementation:**
- Processing Time: {comparison['loops_time']:.3f} seconds  
- MSE: {comparison['loops_mse']:.2f}

⚡ **Performance Gain:**
- Speedup Factor: {comparison['speedup']:.2f}x faster
- Time Saved: {comparison['loops_time'] - comparison['vectorized_time']:.3f} seconds

**Analysis:**
The vectorized implementation uses NumPy's optimized array operations and broadcasting to process multiple grid cells simultaneously, while the loop-based approach processes each cell individually. This demonstrates the significant performance benefits of vectorization in image processing tasks."""
        
        return results_text
        
    except Exception as e:
        error_msg = f"Error in implementation comparison: {str(e)}"
        print(error_msg)
        return error_msg

def download_sample_images():
    """Download sample images if they don't exist."""
    import requests
    
    sample_urls = {
        "sample_landscape.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=500&q=80",
        "sample_nature.jpg": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=500&q=80",
        "sample_animal.jpg": "https://images.unsplash.com/photo-1574158622682-e40e69881006?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=500&q=80"
    }
    
    for filename, url in sample_urls.items():
        if not os.path.exists(filename):
            try:
                print(f"Downloading {filename}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")

def create_demo_images():
    """Load or download sample images for the interface."""
    # Download sample images if they don't exist (for Hugging Face Spaces)
    download_sample_images()
    
    demo_paths = []
    
    # Use the sample images
    sample_files = [
        "sample_landscape.jpg",
        "sample_nature.jpg", 
        "sample_animal.jpg"
    ]
    
    for sample_file in sample_files:
        if os.path.exists(sample_file):
            # Check if file is not empty (some downloads failed)
            if os.path.getsize(sample_file) > 1000:  # At least 1KB
                demo_paths.append(sample_file)
    
    return demo_paths

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .metrics-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=custom_css, title="Image Mosaic Generator", theme=gr.themes.Soft()) as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 Interactive Image Mosaic Generator</h1>
            <p>Transform your images into beautiful mosaics using CIFAR-10 dataset tiles!</p>
            <p><em>Using advanced LAB color space matching and vectorized operations</em></p>
        </div>
        """)
        
        # Main interface
        with gr.Tab("🖼️ Generate Mosaic"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Input controls
                    input_image = gr.Image(
                        label="📤 Upload Your Image",
                        type="pil",
                        height=400
                    )
                    
                    with gr.Row():
                        grid_size = gr.Slider(
                            minimum=8,
                            maximum=64,
                            value=32,
                            step=8,
                            label="🔲 Grid Size (N×N)",
                            info="Higher values create more detailed mosaics but take longer"
                        )
                        
                        show_comparison = gr.Checkbox(
                            label="🔍 Show Side-by-Side Comparison",
                            value=True,
                            info="Display original and mosaic side by side"
                        )
                    
                    generate_btn = gr.Button(
                        "✨ Generate Mosaic",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # Output
                    output_image = gr.Image(
                        label="🎯 Mosaic Result",
                        height=400
                    )
                    
                    with gr.Row():
                        processing_time = gr.Textbox(
                            label="⏱️ Processing Time",
                            interactive=False,
                            scale=1
                        )
                    
                    metrics_display = gr.Markdown(
                        label="📊 Quality Metrics",
                        value="Upload an image and generate a mosaic to see metrics here."
                    )
        
        # Performance Analysis Tab
        with gr.Tab("📈 Performance Analysis"):
            gr.Markdown("""
            ### Performance Analysis Tools
            Analyze how processing time and quality metrics change with different grid sizes,
            and compare vectorized vs loop-based implementations.
            """)
            
            with gr.Row():
                with gr.Column():
                    analysis_image = gr.Image(
                        label="📤 Upload Image for Analysis",
                        type="pil"
                    )
                    
                    with gr.Row():
                        analyze_btn = gr.Button("📊 Analyze Grid Size Performance", variant="secondary")
                        compare_btn = gr.Button("⚡ Compare Implementations", variant="secondary")
                
                with gr.Column():
                    analysis_results = gr.Textbox(
                        label="📋 Analysis Results",
                        lines=15,
                        interactive=False
                    )
            
            performance_plot = gr.Image(
                label="📈 Performance Visualization",
                height=400
            )
            
            # Add sample images for performance analysis too
            gr.Markdown("### 🖼️ Or Use Sample Images for Analysis")
            demo_paths_analysis = create_demo_images()
            
            if demo_paths_analysis:
                gr.Examples(
                    examples=[[path] for path in demo_paths_analysis],
                    inputs=[analysis_image],
                    label="📸 Click to Load Sample Images for Analysis"
                )
        
        
        # Examples section
        gr.Markdown("## 🖼️ Try These Examples")
        gr.Markdown("Click on any example below to load it automatically and see how different image types work with CIFAR-10 tiles:")
        
        # Use the same demo paths that were created for performance analysis
        if demo_paths_analysis:
            # Create examples with different grid sizes for variety
            examples = [
                [demo_paths_analysis[0], 24, True],  # Landscape with medium grid
                [demo_paths_analysis[1], 32, True],  # Nature with standard grid  
                [demo_paths_analysis[2], 16, True]   # Animal with coarse grid
            ]
            gr.Examples(
                examples=examples,
                inputs=[input_image, grid_size, show_comparison],
                label="📸 Sample Images (Landscape, Nature, Animal)"
            )
        else:
            gr.Markdown("*No sample images available*")
        
        # Event handlers
        generate_btn.click(
            fn=generate_mosaic_interface,
            inputs=[input_image, grid_size, show_comparison],
            outputs=[output_image, metrics_display, processing_time]
        )
        
        analyze_btn.click(
            fn=analyze_performance_interface,
            inputs=[analysis_image],
            outputs=[analysis_results, performance_plot]
        )
        
        compare_btn.click(
            fn=compare_implementations_interface,
            inputs=[analysis_image],
            outputs=[analysis_results]
        )
    
    return app

if __name__ == "__main__":
    print("Starting Image Mosaic Generator...")
    print("Generator already initialized and ready to use!")
    
    # Create and launch the app
    app = create_gradio_interface()
    
    # Launch with public sharing enabled
    app.launch(
        share=True,  # Creates public link
        debug=True,
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Use port 7860
        show_error=True
    )
