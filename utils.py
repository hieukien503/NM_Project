import matplotlib.pyplot as plt
import numpy as np
import global_constant as gb

from matplotlib.animation import FuncAnimation

def plot_power_method_convergence(
    vectors,
    eigenvalues
):

    # 2D Plot: Convergence of differences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(eigenvalues, marker='o', linestyle='-', color='blue', label='Difference')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Dominant Eigenvalue')
    ax1.set_yscale('log')
    ax1.set_title(f'Convergence of Dominant Eigenvalue using Power Method')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()

    # 3D Plot: Evolution of orthogonal vectors
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(vectors[:, 0], vectors[:, 1], vectors[:, 2], marker='o', color='blue', label='Eigenvector Path', linestyle='-', linewidth=2)
    ax2.scatter(vectors[0, 0], vectors[0, 1], vectors[0, 2], color='red', s=100, label='Start')
    ax2.scatter(vectors[-1, 0], vectors[-1, 1], vectors[-1, 2], color='green', s=100, label='Converged')
    ax2.set_title(f"Convergence of Dominant Eigenvector using Power Method")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_QR_algorithm_convergence(
    matrices,
):
    fig, ax1 = plt.subplots(figsize=(6, 6))
    cax = ax1.matshow(matrices[0], cmap='viridis', vmin=0)
    fig.colorbar(cax, ax=ax1)

    # Function to update the plot
    def update(iteration):
        ax1.clear()
        cax = ax1.matshow(matrices[iteration], cmap='viridis', vmin=0)
        ax1.set_title(f'QR Iteration {iteration}')
        return [cax]

    # Animation
    ani = FuncAnimation(fig, update, frames=min(len(matrices), gb.max_frame), interval=50, blit=False, repeat=False)
    plt.show()

def load_matrix(filename: str) -> np.ndarray:
    """
        The input file contains n lines of floating-point numbers
        At each line, the numbers are separated by space
        No newline exists at the end of the last line
    """
    with open(file=filename) as f:
        lines = f.readlines()
        lines = [line if line[-1] != '\n' else line[:-1] for line in lines]
        matrix = []
        for line in lines:
            matrix.append([float(item) for item in line.split(' ')])
        
        return np.array(matrix)

def gen_matrix(filename: str, low: int, high: int, maxsize: int = 5):
    if low >= high:
        raise ValueError("`low` must smaller than `high`")
    
    with open(file=filename, mode='+w') as f:
        A = np.random.uniform(low, high, size=(maxsize, maxsize)).tolist()
        lines = [" ".join([str(x) for x in item]) for item in A]
        for index in range(len(lines) - 1):
            lines[index] += '\n'
        
        f.writelines(lines)

def gen_sym_matrix(filename: str, low: int, high: int, maxsize: int = 5):
    if low >= high:
        raise ValueError("`low` must smaller than `high`")
    
    with open(file=filename, mode='+w') as f:
        A = np.random.uniform(-200, 200, size=(maxsize, maxsize))
        A = (A + A.T) / 2
        A = A.tolist()
        lines = [" ".join([str(x) for x in item]) for item in A]
        for index in range(len(lines) - 1):
            lines[index] += '\n'
        
        f.writelines(lines)

def print_matrix(matrix: np.ndarray):
    print("\n".join([" ".join([f"{item:.6f}" for item in sublist]) for sublist in matrix]))

def print_eigens(eigenvalues, eigenvectors):
    if len(eigenvalues) != len(eigenvectors):
        raise ValueError("`eigenvalues` and `eigenvectors` must have the same length")
    
    print(f'{"Eigenvalue":<20} {"Eigenvector":<20}')
    for val, vec in zip(eigenvalues, eigenvectors.T):
        vec_str = ', '.join(f'{x:.6f}' for x in vec)
        print(f'{val:<20.6f} {vec_str:<20}')