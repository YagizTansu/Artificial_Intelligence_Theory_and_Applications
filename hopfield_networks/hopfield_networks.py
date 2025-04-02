import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import random

class HopfieldNetwork:
    def __init__(self, size):
        """
        Initialize a Hopfield network with a given number of neurons.
        
        Args:
            size: Number of neurons in the network
        """
        self.size = size
        self.weights = np.zeros((size, size))
        self.state = np.zeros(size)
        self.stored_patterns = []
        
    def train(self, patterns):
        """
        Train the network using the Hebbian learning rule to store patterns.
        
        Args:
            patterns: List of patterns to store, each pattern should be a 1D array of -1/1 values
        """
        self.stored_patterns = patterns.copy()
        
        # Reset weights
        self.weights = np.zeros((self.size, self.size))
        
        # Apply Hebbian learning rule
        for pattern in patterns:
            # Adjust values from 0/1 to -1/1 if needed
            pattern_values = np.copy(pattern)
            if np.all((pattern_values == 0) | (pattern_values == 1)):
                pattern_values = 2 * pattern_values - 1
                
            # Outer product for weight update
            self.weights += np.outer(pattern_values, pattern_values)
            
        # Normalize by number of patterns
        self.weights /= self.size
        
        # Set diagonal elements to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)
        
    def update_neuron(self, index):
        """
        Update a single neuron based on the weighted sum of inputs.
        
        Args:
            index: The index of the neuron to update
            
        Returns:
            True if the neuron changed state, False otherwise
        """
        h = np.dot(self.weights[index], self.state)
        prev_state = self.state[index]
        
        # Apply threshold function
        if h > 0:
            self.state[index] = 1
        elif h < 0:
            self.state[index] = -1
        # h == 0: state remains unchanged
        
        return prev_state != self.state[index]
    
    def update_async(self, max_iterations=100):
        """
        Asynchronously update neurons until convergence or max iterations.
        
        Args:
            max_iterations: Maximum number of complete network updates
            
        Returns:
            Number of iterations performed and energy history
        """
        iterations = 0
        energy_history = []
        
        for _ in range(max_iterations):
            changed = False
            
            # Update neurons in random order
            indices = list(range(self.size))
            random.shuffle(indices)
            
            for idx in indices:
                changed = self.update_neuron(idx) or changed
            
            # Store energy
            energy_history.append(self.energy())
            iterations += 1
            
            # If no neuron changed state, the network has converged
            if not changed:
                break
                
        return iterations, energy_history
    
    def update_sync(self, max_iterations=100):
        """
        Synchronously update all neurons until convergence or max iterations.
        
        Args:
            max_iterations: Maximum number of updates
            
        Returns:
            Number of iterations performed and energy history
        """
        iterations = 0
        energy_history = []
        
        for _ in range(max_iterations):
            # Calculate all new states based on current states
            h = np.dot(self.weights, self.state)
            new_state = np.where(h > 0, 1, np.where(h < 0, -1, self.state))
            
            # Check if any neurons changed state
            changed = not np.array_equal(self.state, new_state)
            
            # Update all neurons simultaneously
            self.state = new_state
            
            # Store energy
            energy_history.append(self.energy())
            iterations += 1
            
            # If no neuron changed state, the network has converged
            if not changed:
                break
                
        return iterations, energy_history
    
    def recall(self, pattern, update_mode="async", max_iterations=100):
        """
        Recall a stored pattern from a given input pattern.
        
        Args:
            pattern: Input pattern
            update_mode: 'async' for asynchronous update or 'sync' for synchronous
            max_iterations: Maximum number of updates
            
        Returns:
            Recalled pattern, number of iterations, and energy history
        """
        # Convert pattern to -1/1 if needed
        if np.all((pattern == 0) | (pattern == 1)):
            self.state = 2 * pattern - 1
        else:
            self.state = pattern.copy()
        
        # Update the network
        if update_mode == "async":
            iterations, energy_history = self.update_async(max_iterations)
        else:
            iterations, energy_history = self.update_sync(max_iterations)
            
        return self.state, iterations, energy_history
    
    def energy(self):
        """
        Calculate the energy of the current network state.
        
        Returns:
            Energy value
        """
        return -0.5 * np.dot(np.dot(self.state, self.weights), self.state)
    
    def add_noise(self, pattern, noise_level=0.1):
        """
        Add random noise to a pattern.
        
        Args:
            pattern: Input pattern
            noise_level: Fraction of bits to flip (between 0 and 1)
            
        Returns:
            Noisy pattern
        """
        noisy_pattern = pattern.copy()
        num_flips = int(self.size * noise_level)
        flip_indices = np.random.choice(self.size, num_flips, replace=False)
        
        for idx in flip_indices:
            noisy_pattern[idx] = -noisy_pattern[idx]
            
        return noisy_pattern


class HopfieldVisualizer:
    def __init__(self, network, grid_size):
        """
        Initialize the visualizer for a Hopfield network.
        
        Args:
            network: HopfieldNetwork instance
            grid_size: Tuple (width, height) for visualization
        """
        self.network = network
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
    def reshape_pattern(self, pattern):
        """
        Reshape a 1D pattern into a 2D grid for visualization.
        """
        return pattern.reshape(self.grid_size)
    
    def visualize_pattern(self, pattern, title=""):
        """
        Visualize a pattern as a grid.
        """
        # Convert to -1/1 if needed
        if np.all((pattern == 0) | (pattern == 1)):
            pattern_values = 2 * pattern - 1
        else:
            pattern_values = pattern.copy()
        
        # Reshape if needed
        if pattern_values.ndim == 1:
            grid_pattern = self.reshape_pattern(pattern_values)
        else:
            grid_pattern = pattern_values
            
        # Create a custom colormap: -1 = white, 1 = black
        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [-1.5, 0, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Show the grid
        self.ax.clear()
        self.ax.imshow(grid_pattern, cmap=cmap, norm=norm, interpolation='none')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(title)
        plt.pause(0.1)
        
    def animate_recall(self, pattern, update_mode="async", max_iterations=100, interval=200):
        """
        Animate the recall process.
        """
        # Convert pattern to -1/1 if needed
        if np.all((pattern == 0) | (pattern == 1)):
            pattern_values = 2 * pattern - 1
        else:
            pattern_values = pattern.copy()
            
        # Initialize network state
        self.network.state = pattern_values.copy()
        
        # Set up animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Create custom colormap for the grid
        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [-1.5, 0, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        energy_values = []
        iterations = []
        
        def init():
            grid_pattern = self.reshape_pattern(self.network.state)
            img = ax1.imshow(grid_pattern, cmap=cmap, norm=norm, interpolation='none')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(f"Network State (Iteration 0)")
            
            energy = self.network.energy()
            energy_values.append(energy)
            iterations.append(0)
            
            ax2.plot(iterations, energy_values, 'b-')
            ax2.set_xlabel('Iterations')
            ax2.set_ylabel('Energy')
            ax2.set_title('Energy Function')
            
            return img,
        
        def update(frame):
            if update_mode == "async":
                # Update a random neuron
                idx = np.random.randint(0, self.network.size)
                self.network.update_neuron(idx)
            else:
                # Update all neurons synchronously
                h = np.dot(self.network.weights, self.network.state)
                self.network.state = np.where(h > 0, 1, np.where(h < 0, -1, self.network.state))
            
            # Update the visualization
            grid_pattern = self.reshape_pattern(self.network.state)
            img = ax1.imshow(grid_pattern, cmap=cmap, norm=norm, interpolation='none')
            ax1.set_title(f"Network State (Iteration {frame+1})")
            
            # Update energy plot
            energy = self.network.energy()
            energy_values.append(energy)
            iterations.append(frame+1)
            
            ax2.clear()
            ax2.plot(iterations, energy_values, 'b-')
            ax2.set_xlabel('Iterations')
            ax2.set_ylabel('Energy')
            ax2.set_title('Energy Function')
            
            return img,
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=range(max_iterations),
                                     init_func=init, blit=False, interval=interval)
        
        plt.tight_layout()
        return ani
    
    def show_patterns(self, patterns, title="Stored Patterns"):
        """
        Display multiple patterns in a grid.
        """
        n = len(patterns)
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        
        # Create a custom colormap: -1 = white, 1 = black
        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [-1.5, 0, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        for i, pattern in enumerate(patterns):
            if rows == 1 and cols == 1:
                ax = axs
            elif rows == 1 or cols == 1:
                ax = axs[i]
            else:
                ax = axs[i // cols, i % cols]
                
            # Convert to -1/1 if needed
            if np.all((pattern == 0) | (pattern == 1)):
                pattern_values = 2 * pattern - 1
            else:
                pattern_values = pattern
                
            grid_pattern = self.reshape_pattern(pattern_values)
            ax.imshow(grid_pattern, cmap=cmap, norm=norm, interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
            
        # Hide any unused subplots
        for i in range(n, rows * cols):
            if rows == 1 and cols == 1:
                continue
            elif rows == 1 or cols == 1:
                axs[i].axis('off')
            else:
                axs[i // cols, i % cols].axis('off')
                
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


class HopfieldDemo:
    def __init__(self):
        """
        Setup the Hopfield network demo
        """
        # Default parameters
        self.grid_width = 10
        self.grid_height = 10
        self.network = HopfieldNetwork(self.grid_width * self.grid_height)
        self.visualizer = HopfieldVisualizer(self.network, (self.grid_height, self.grid_width))
        
        # Sample patterns
        self.default_patterns = self.create_default_patterns()
        
    def create_default_patterns(self):
        """
        Create some default patterns for demonstration.
        """
        patterns = []
        shape = (self.grid_height, self.grid_width)
        
        # Pattern 1: X shape
        p1 = np.zeros(shape)
        for i in range(shape[0]):
            p1[i, i] = 1
            p1[i, shape[1]-i-1] = 1
        
        # Pattern 2: + shape
        p2 = np.zeros(shape)
        mid_row = shape[0] // 2
        mid_col = shape[1] // 2
        p2[mid_row, :] = 1
        p2[:, mid_col] = 1
        
        # Pattern 3: O shape
        p3 = np.zeros(shape)
        border = 1
        p3[border:-border, border] = 1
        p3[border:-border, -border-1] = 1
        p3[border, border:-border] = 1
        p3[-border-1, border:-border] = 1
        
        # Pattern 4: Z shape
        p4 = np.zeros(shape)
        p4[0, :] = 1
        p4[-1, :] = 1
        for i in range(shape[0]):
            p4[i, shape[1]-i-1] = 1
            
        # Convert to 1D arrays
        patterns.append(p1.flatten())
        patterns.append(p2.flatten())
        patterns.append(p3.flatten())
        patterns.append(p4.flatten())
        
        return [2 * p - 1 for p in patterns]  # Convert to -1/1
    
    def run(self):
        """
        Run the interactive Hopfield network demo
        """
        print("Welcome to the Hopfield Network Demo!")
        print("====================================")
        
        # Train the network with default patterns
        self.network.train(self.default_patterns)
        
        # Show the stored patterns
        self.visualizer.show_patterns(self.default_patterns, "Stored Patterns")
        
        while True:
            print("\nChoose an option:")
            print("1. Show stored patterns")
            print("2. Test recall with noise")
            print("3. Create custom pattern")
            print("4. Train with custom patterns")
            print("5. Animate recall process")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                self.visualizer.show_patterns(self.default_patterns, "Stored Patterns")
                
            elif choice == '2':
                self.test_recall_with_noise()
                
            elif choice == '3':
                self.create_custom_pattern()
                
            elif choice == '4':
                self.train_custom_patterns()
                
            elif choice == '5':
                self.animate_recall()
                
            elif choice == '6':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please try again.")
    
    def test_recall_with_noise(self):
        """
        Test recall with a noisy version of a stored pattern
        """
        pattern_idx = int(input(f"Select a pattern (1-{len(self.default_patterns)}): ")) - 1
        if pattern_idx < 0 or pattern_idx >= len(self.default_patterns):
            print("Invalid pattern index")
            return
            
        pattern = self.default_patterns[pattern_idx]
        
        noise_level = float(input("Enter noise level (0.0-1.0): "))
        if noise_level < 0 or noise_level > 1:
            print("Invalid noise level")
            return
            
        noisy_pattern = self.network.add_noise(pattern, noise_level)
        
        update_mode = input("Update mode (async/sync): ").lower()
        if update_mode not in ['async', 'sync']:
            update_mode = 'async'  # Default
            
        # Show original and noisy patterns
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Create a custom colormap
        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [-1.5, 0, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Original pattern
        grid_original = noisy_pattern.reshape((self.grid_height, self.grid_width))
        axs[0].imshow(grid_original, cmap=cmap, norm=norm, interpolation='none')
        axs[0].set_title("Noisy Pattern")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        # Recall from network
        recalled_pattern, iterations, energy_history = self.network.recall(noisy_pattern, update_mode)
        grid_recalled = recalled_pattern.reshape((self.grid_height, self.grid_width))
        
        axs[1].imshow(grid_recalled, cmap=cmap, norm=norm, interpolation='none')
        axs[1].set_title(f"Recalled Pattern (after {iterations} iterations)")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        plt.suptitle(f"Recall Test with {noise_level*100:.1f}% Noise")
        plt.tight_layout()
        plt.show()
        
        # Plot energy function
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(energy_history)), energy_history, 'b-')
        plt.xlabel('Iterations')
        plt.ylabel('Energy')
        plt.title('Energy Function During Recall')
        plt.grid(True)
        plt.show()
        
    def create_custom_pattern(self):
        """
        Create a custom pattern using an interactive grid
        """
        # Initialize an empty pattern
        pattern = np.zeros((self.grid_height, self.grid_width))
        
        # Create an interactive figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create a custom colormap: -1 = white, 1 = black
        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [-1.5, 0, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Display initial empty pattern
        img = ax.imshow(pattern, cmap=cmap, norm=norm, interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Click to toggle cells (close figure when finished)")
        
        # Event handler for mouse clicks
        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return
            
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                # Toggle cell value
                pattern[y, x] = 1 if pattern[y, x] == 0 else 0
                
                # Update display
                img.set_data(pattern)
                fig.canvas.draw_idle()
                
        # Connect the event handler
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Show the figure
        plt.show()
        
        # Convert to 1D pattern with -1/1 values
        custom_pattern = (2 * pattern - 1).flatten()
        
        # Show the final pattern
        plt.figure(figsize=(6, 6))
        plt.imshow(pattern, cmap=cmap, norm=norm, interpolation='none')
        plt.title("Custom Pattern")
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
        # Ask if the user wants to save this pattern
        save_choice = input("Save this pattern? (y/n): ")
        if save_choice.lower() == 'y':
            self.default_patterns.append(custom_pattern)
            print("Pattern added to the list")
            
        return custom_pattern
    
    def train_custom_patterns(self):
        """
        Train the network with custom patterns
        """
        # Show current stored patterns
        self.visualizer.show_patterns(self.default_patterns, "Current Stored Patterns")
        
        # Option to clear existing patterns
        clear_choice = input("Clear existing patterns before training? (y/n): ")
        if clear_choice.lower() == 'y':
            self.default_patterns = []
            
        # Option to add new patterns
        add_patterns = True
        while add_patterns:
            custom_pattern = self.create_custom_pattern()
            
            if len(self.default_patterns) >= 1:
                add_more = input("Add another pattern? (y/n): ")
                if add_more.lower() != 'y':
                    add_patterns = False
        
        # Train the network with the patterns
        if len(self.default_patterns) > 0:
            self.network.train(self.default_patterns)
            print(f"Network trained with {len(self.default_patterns)} patterns")
            self.visualizer.show_patterns(self.default_patterns, "Stored Patterns")
        else:
            print("No patterns to train with")
    
    def animate_recall(self):
        """
        Animate the recall process
        """
        # Choose a pattern
        pattern_idx = int(input(f"Select a pattern (1-{len(self.default_patterns)}): ")) - 1
        if pattern_idx < 0 or pattern_idx >= len(self.default_patterns):
            print("Invalid pattern index")
            return
            
        pattern = self.default_patterns[pattern_idx]
        
        # Ask for noise level
        noise_level = float(input("Enter noise level (0.0-1.0): "))
        if noise_level < 0 or noise_level > 1:
            print("Invalid noise level")
            return
            
        noisy_pattern = self.network.add_noise(pattern, noise_level)
        
        # Ask for update mode
        update_mode = input("Update mode (async/sync): ").lower()
        if update_mode not in ['async', 'sync']:
            update_mode = 'async'  # Default
            
        # Show the noisy pattern
        self.visualizer.visualize_pattern(noisy_pattern, "Noisy Pattern")
        
        # Ask for animation settings
        max_iterations = int(input("Maximum iterations for animation: "))
        interval = int(input("Animation interval in ms (e.g., 200): "))
        
        # Create and display animation
        ani = self.visualizer.animate_recall(noisy_pattern, update_mode, max_iterations, interval)
        plt.show()


# Run the demo when script is executed
if __name__ == "__main__":
    demo = HopfieldDemo()
    demo.run()
