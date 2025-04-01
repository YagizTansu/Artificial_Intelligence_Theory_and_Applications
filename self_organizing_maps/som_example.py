import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Set random seed for reproducibility
np.random.seed(42)

# Function to create dataset
def create_dataset(dataset_type='random'):
    """
    Create and preprocess a dataset for SOM training.
    
    Parameters:
        dataset_type (str): Type of dataset ('random' or 'iris')
        
    Returns:
        tuple: (scaled_data, labels)
    """
    if dataset_type == 'random':
        # Generate random data points
        data = np.random.rand(300, 2)
        labels = np.zeros(300)  # No real labels for random data
    elif dataset_type == 'iris':
        # Load Iris dataset
        iris = load_iris()
        data = iris.data
        labels = iris.target
    else:
        raise ValueError("Dataset type must be 'random' or 'iris'")
    
    # Scale data to [0,1] range
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, labels

# Function to initialize SOM
def initialize_som(data, grid_size=10, sigma=1.0, learning_rate=0.5):
    """
    Initialize a Self-Organizing Map.
    
    Parameters:
        data (ndarray): Input data to determine dimensions
        grid_size (int): Size of the grid (square SOM)
        sigma (float): Radius of neighborhood function
        learning_rate (float): Initial learning rate
    
    Returns:
        MiniSom: Initialized SOM object
    """
    # Initialize SOM
    som_dim = grid_size
    input_dim = data.shape[1]
    
    # Define a safe decay function that handles edge cases
    def safe_decay(x, max_iter, t):
        # Prevent division by zero
        if max_iter <= 0:
            return x
        # Ensure max_iter is large enough to prevent division by very small numbers
        safe_max = max(max_iter, 1.0)
        # Apply decay formula with protection against division by zero
        return x / (1.0 + t / (safe_max / 2.0))
    
    som = MiniSom(som_dim, som_dim, input_dim, 
                  sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian',
                  decay_function=safe_decay)
    
    # Initialize weights randomly
    som.random_weights_init(data)
    
    return som

# Function to get grid position of Best Matching Unit (BMU)
def get_bmu_pos(som, x):
    """Get position of Best Matching Unit for input x"""
    return som.winner(x)

# Function to compute the U-Matrix (Unified distance matrix)
def compute_u_matrix(som, data):
    """
    Compute the U-Matrix (Unified Distance Matrix) for the SOM.
    
    Parameters:
        som (MiniSom): Trained SOM
        data (ndarray): Input data
        
    Returns:
        ndarray: U-Matrix values
    """
    weights = som.get_weights()
    u_matrix = np.zeros((weights.shape[0], weights.shape[1]))
    
    # For each node, compute the average distance to its neighbors
    for x in range(weights.shape[0]):
        for y in range(weights.shape[1]):
            neighbors_weights = []
            
            # Get the weights of the neighboring nodes (more efficient approach)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < weights.shape[0] and 0 <= ny < weights.shape[1]:
                    neighbors_weights.append(weights[nx, ny, :])
            
            if neighbors_weights:  # Check if there are any neighbors
                neighbors_weights = np.array(neighbors_weights)
                dist = np.linalg.norm(weights[x, y, :] - neighbors_weights, axis=1).mean()
                u_matrix[x, y] = dist
            
    return u_matrix

# Function to visualize SOM training in real-time
def visualize_som_training(data, labels=None, grid_size=10, max_iter=1000, 
                          sigma=1.0, learning_rate=0.5, dataset_type='random'):
    """
    Visualize SOM training in real-time with animation.
    
    Parameters:
        data (ndarray): Input data for training
        labels (ndarray, optional): Labels for the data points
        grid_size (int): Size of the SOM grid
        max_iter (int): Maximum number of training iterations
        sigma (float): Initial neighborhood radius
        learning_rate (float): Initial learning rate
        dataset_type (str): Type of dataset ('random' or 'iris')
        
    Returns:
        tuple: (trained_som, animation)
    """
    # Create figure and subplots
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Data scatter plot
    ax2 = fig.add_subplot(gs[0, 1])  # SOM grid
    ax3 = fig.add_subplot(gs[0, 2])  # U-Matrix
    ax4 = fig.add_subplot(gs[1, :])  # Animation of training
    
    # Initialize SOM
    som = initialize_som(data, grid_size=grid_size, sigma=sigma, learning_rate=learning_rate)
    
    # Plot original data
    if dataset_type == 'iris':
        scatter = ax1.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        ax1.set_title('Original Data (First 2 features of Iris)')
        # Add color bar for labels
        plt.colorbar(scatter, ax=ax1, label='Species')
    else:
        scatter = ax1.scatter(data[:, 0], data[:, 1], c='blue')
        ax1.set_title('Random Data Distribution')
    
    # Create grid for SOM
    x_grid, y_grid = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    
    # Initial U-Matrix
    u_matrix = compute_u_matrix(som, data)
    u_matrix_img = ax3.imshow(u_matrix, cmap='viridis')
    plt.colorbar(u_matrix_img, ax=ax3, label='Distance')
    ax3.set_title('U-Matrix (Unified Distance)')
    
    # Animation setup
    weights_grid = np.zeros((grid_size, grid_size, 2))
    grid_plot = ax2.scatter(weights_grid[:,:,0].flatten(), 
                           weights_grid[:,:,1].flatten(),
                           s=50, c='gray', marker='s')
    ax2.set_title('SOM Grid')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    # Training animation setup
    step_text = ax4.text(0.5, 0.5, '', ha='center', va='center', fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # Extract the first 2 dimensions for visualization
    if data.shape[1] > 2:
        projection = np.copy(som.get_weights()[:,:,:2])
    else:
        projection = np.copy(som.get_weights())
    
    # Plot initial weights
    grid_img = ax2.scatter(projection[:,:,0].flatten(), 
                         projection[:,:,1].flatten(), 
                         s=50, c='red', marker='o')
    
    # Connect grid points with lines
    line_segments = []
    for i in range(grid_size):
        line, = ax2.plot(projection[i,:,0], projection[i,:,1], 'gray', alpha=0.5)
        line_segments.append(line)
    
    for j in range(grid_size):
        line, = ax2.plot(projection[:,j,0], projection[:,j,1], 'gray', alpha=0.5)
        line_segments.append(line)
    
    # Keep track of the total iterations performed
    total_iterations_done = [0]
    training_complete = [False]
    
    # Calculate current learning rate and sigma based on iteration
    def get_current_parameters(initial_lr, initial_sigma, iteration, max_iter):
        """Calculate current learning rate and sigma with decay"""
        # Common decay function with protection against division by zero
        if max_iter <= 0:
            return initial_lr, initial_sigma
        
        safe_max = max(max_iter, 1.0)
        current_lr = initial_lr / (1.0 + iteration / (safe_max / 2.0))
        current_sigma = initial_sigma / (1.0 + iteration / (safe_max / 2.0))
        return current_lr, current_sigma
    
    # Animation update function
    def update(frame):
        # Only update if training is not complete
        if not training_complete[0]:
            # Train SOM for a batch of iterations - slower for better visualization
            iterations_per_frame = max(1, max_iter // 200) if max_iter > 0 else 1
            
            # Calculate how many iterations we should do in this frame
            remaining_iters = max(0, max_iter - total_iterations_done[0])
            iters_this_frame = min(iterations_per_frame, remaining_iters)
            
            if iters_this_frame > 0:
                for i in range(iters_this_frame):
                    # Pick a random sample
                    idx = np.random.randint(len(data))
                    current_iter = total_iterations_done[0]
                    
                    try:
                        # Calculate current learning rate and sigma
                        current_lr, current_sigma = get_current_parameters(
                            learning_rate, sigma, current_iter, max_iter)
                        
                        # Update the SOM weights
                        winner = som.winner(data[idx])
                        som.update(data[idx], winner, current_iter, max(1, max_iter))  # Ensure max_iter is at least 1
                        
                        # Increment the iteration counter
                        total_iterations_done[0] += 1
                    except Exception as e:
                        print(f"Error during update: {e}")
                        training_complete[0] = True
                        break
                
                # Update U-Matrix
                u_matrix = compute_u_matrix(som, data)
                u_matrix_img.set_array(u_matrix)
                
                # Update weights visualization
                weights = som.get_weights()
                if data.shape[1] > 2:
                    projection = np.copy(weights[:,:,:2])
                else:
                    projection = np.copy(weights)
                    
                grid_img.set_offsets(np.column_stack([projection[:,:,0].flatten(), 
                                                    projection[:,:,1].flatten()]))
                
                # Update grid lines
                for i in range(grid_size):
                    line_segments[i].set_data(projection[i,:,0], projection[i,:,1])
                
                for j in range(grid_size):
                    line_segments[grid_size + j].set_data(projection[:,j,0], projection[:,j,1])
                
                progress = total_iterations_done[0] / max_iter * 100
                
                # Get current learning rate and sigma
                current_lr, current_sigma = get_current_parameters(
                    learning_rate, sigma, total_iterations_done[0], max_iter)
                
                param_text = f"Learning Rate: {current_lr:.4f}, Sigma: {current_sigma:.4f}"
                
                step_text.set_text(f"Training Progress: {progress:.1f}% \n"
                                   f"Iteration: {total_iterations_done[0]}/{max_iter}\n"
                                   f"{param_text}")
                
                # Check if we've reached the maximum iterations
                if total_iterations_done[0] >= max_iter:
                    training_complete[0] = True
                    step_text.set_text(f"Training Complete: 100% \n"
                                      f"Iteration: {max_iter}/{max_iter}\n"
                                      f"{param_text}\n\nClick close to view final results")
            
        return [grid_img, u_matrix_img, step_text] + line_segments
    
    # Create animation with slower interval for better visualization
    ani = FuncAnimation(fig, update, frames=210, interval=100, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()
    
    return som, ani

# Function to visualize final SOM results
def visualize_final_som(som, data, labels=None, dataset_type='random'):
    """
    Visualize final trained SOM with various plots.
    
    Parameters:
        som (MiniSom): Trained SOM
        data (ndarray): Input data
        labels (ndarray, optional): Labels for the data points
        dataset_type (str): Type of dataset ('random' or 'iris')
    """
    weights = som.get_weights()
    grid_size = weights.shape[0]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # 1. Plot U-Matrix
    plt.subplot(2, 2, 1)
    u_matrix = compute_u_matrix(som, data)
    u_img = plt.imshow(u_matrix, cmap='viridis')
    plt.colorbar(u_img)
    plt.title('U-Matrix (Unified Distance Matrix)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 2. Plot component planes (for higher dimensional data)
    if data.shape[1] > 2:
        for i in range(min(4, data.shape[1])):  # Show up to 4 components
            plt.subplot(2, 4, i+5)
            component_plane = weights[:,:,i]
            cp_img = plt.imshow(component_plane, cmap='viridis')
            plt.colorbar(cp_img)
            comp_title = f'Feature {i+1}'
            if dataset_type == 'iris' and i < len(load_iris().feature_names):
                comp_title += f" ({load_iris().feature_names[i]})"
            plt.title(comp_title)
            plt.xlabel('X')
            plt.ylabel('Y')
    
    # 3. Plot data points on SOM grid (density map)
    plt.subplot(2, 2, 3)
    
    # Get position of each data point on SOM
    x_coords = np.zeros(data.shape[0])
    y_coords = np.zeros(data.shape[0])
    
    for i, x in enumerate(data):
        w = som.winner(x)
        x_coords[i], y_coords[i] = w[0], w[1]
    
    # Create a map for the points
    map_size = weights.shape[:2]
    mapped = np.zeros(map_size)
    
    # Count points in each node
    for x, y in zip(x_coords, y_coords):
        mapped[int(x), int(y)] += 1
    
    density_img = plt.imshow(mapped, cmap='Blues')
    plt.colorbar(density_img, label='Number of samples')
    plt.title('Data Density on SOM')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 4. Visualization of clustered data
    plt.subplot(2, 2, 4)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if dataset_type == 'iris' and labels is not None:
        # For labeled data, show cluster assignments
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(x_coords[mask], y_coords[mask], 
                       c=[colors[i]], alpha=0.7,
                       label=f'Class {label}')
        plt.legend()
        plt.title('Data Points Mapped to SOM (colored by true labels)')
    else:
        # For random data, just show density
        plt.scatter(x_coords, y_coords, c='blue', alpha=0.5)
        plt.title('Data Points Mapped to SOM')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    """Main function to execute SOM training and visualization"""
    # Process command-line arguments if needed
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Organizing Map Visualization')
    parser.add_argument('--dataset', type=str, default='iris', 
                       choices=['random', 'iris'], 
                       help='Dataset to use (random or iris)')
    parser.add_argument('--grid_size', type=int, default=10,
                       help='Size of the SOM grid')
    parser.add_argument('--iterations', type=int, default=100000,
                       help='Number of training iterations')
    
    args = parser.parse_args()
    
    # Create dataset
    dataset_type = args.dataset
    data, labels = create_dataset(dataset_type)
    
    print(f"Dataset: {dataset_type}")
    print(f"Dataset shape: {data.shape}")
    
    # Set SOM parameters with validation
    grid_size = max(2, args.grid_size)  # Ensure grid size is at least 2
    max_iter = max(100, args.iterations)  # Ensure at least 100 iterations
    sigma = 2.0  # Increased initial sigma for better neighborhood effect
    learning_rate = 0.8  # Increased initial learning rate
    
    # Train and visualize SOM
    print("Training SOM with real-time visualization...")
    try:
        som, _ = visualize_som_training(
            data, labels, 
            grid_size=grid_size, 
            max_iter=max_iter,
            sigma=sigma, 
            learning_rate=learning_rate,
            dataset_type=dataset_type
        )
        
        # Show final results
        print("Displaying final SOM visualization...")
        visualize_final_som(som, data, labels, dataset_type)
    
    except Exception as e:
        print(f"Error during SOM training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
