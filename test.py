from sympy import *
from scipy.linalg import null_space
from utils import load_matrix, print_eigens, print_matrix, plot_power_method_convergence
from qr_algorithm import *

import numpy as np
import global_constant as gb
import time

def characteristics_method(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenvalues and eigenvectors of matrix A using the characteristic polynomial method."""
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    
    n = A.shape[1]
    A_sym = Matrix(A.tolist()).applyfunc(Rational)
    lam = symbols('lambda')
    # I = Matrix.eye(A_sym.shape[0])
    # char_matrix = A_sym - lam * I
    # char_poly = det(char_matrix).simplify()                   # Setup the characteristics polynomial det(A - lambda.I)
    # solutions = solve(char_poly, lam)                         # Solve for lambda

    # char_poly = A_sym.charpoly(lam).as_expr()                 # Calculate the characteristics polynomial
    # solutions = solve(char_poly, lam)                         # Solve for lambda (exact solution, symbolically)

    char_poly = A_sym.charpoly(lam).as_expr()                 # Calculate the characteristics polynomial
    solutions = nroots(
        char_poly,
        n=15,
        maxsteps=gb.args.test_maxiter
    )                                                         # Solve for lambda (approximate solution, numerically)
    solutions = [val.evalf() for val in solutions]              # Evaluate the solutions to floating-point numbers
    solutions = np.array(solutions, dtype=np.complex64)         # Convert the solutions to complex numbers
    eigenvalues = []
    eigenvectors = []
    for eigenvalue in solutions:
        eig_eq = A - eigenvalue * np.eye(n)
        nullspace = null_space(eig_eq, rcond=gb.args.test_tol)
        if nullspace.size > 0:
            eigenvectors.extend(vec for vec in nullspace.T)
            eigenvalues.extend(eigenvalue for _ in nullspace.T)
    
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors).T
    return eigenvalues, eigenvectors

def generate_eig_sympy(A: np.ndarray):
    def iszerofunc(x):
        import warnings
        result = x.rewrite(exp).simplify().is_zero
        if result is None:
            warnings.warn(f"File {__name__}.py: Zero testing for {x} result in None")
        
        return result
    
    eigen_data = Matrix(A).eigenvects(iszerofunc=iszerofunc)
    eigenvalues, eigenvectors = [], []
    for ev, multiply, evec in eigen_data:
        eigenvectors.extend(vec.flat() for vec in evec)
        eigenvalues.extend(ev for _ in range(multiply))
    
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors).T
    return eigenvalues, eigenvectors

def generate_eig_numpy(A: np.ndarray):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors.T

def QR_decompostion(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Q, R = np.linalg.qr(A)
    return Q, R

def power_method(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Compute the dominant eigenvalue and eigenvector of matrix A using the power method."""

    n, _ = A.shape
    b_k = np.random.rand(n)

    for _ in range(max_iter):
        # Matrix-vector multiplication
        b_k1 = A @ b_k

        # Normalize the resulting vector
        b_k1_norm = np.linalg.norm(b_k1)
        if b_k1_norm == 0:
            raise ValueError("Encountered zero vector during iteration.")
        b_k1 = b_k1 / b_k1_norm

        # Check for convergence
        if np.linalg.norm(b_k1 - b_k) < tol:
            break

        b_k = b_k1

    # Rayleigh quotient for eigenvalue approximation
    eigenvalue = (b_k.T @ A @ b_k) / (b_k.T @ b_k)
    eigenvector = b_k
    return np.array([eigenvalue]), eigenvector

class TestCase:
    def __init__ (self, filename: str):
        self.A = load_matrix(filename)
    
    def test_eigen(self):
        methods = ["QR Algorithm", "QR Algorithm with Wilkinson Shift", "Francis Double Shift QR"]

        if self.A.shape[0] < 10:
            print("Matrix A:")
            print_matrix(self.A)
        
        A1 = self.A.copy()
        char_poly_start = time.time()
        eigenvalues, eigenvectors = characteristics_method(A1)
        char_poly_end = time.time()
        if len(eigenvalues) < 10 and len(eigenvectors) < 10:
            print("Eigenvalues and Eigenvectors")
            print("Using CharPoly Method:")
            print_eigens(eigenvalues, eigenvectors)

        print(f"Time to find eigenvalues and eigenvectors using CharPoly Method: {(char_poly_end - char_poly_start):.4f} seconds")
        
        for idx, method in enumerate([qr_algorithm, qr_algorithm_wilkinson, francis_double_shift_qr]):
            A2 = self.A.copy()
            qr_algo_start = time.time()
            eigenvals, eigenvecs = method(A2, gb.args.test_maxiter, gb.args.test_tol)
            qr_algo_end = time.time()
            
            if len(eigenvals) < 10 and len(eigenvecs) < 10:
                print(f"Using QR Algorithm with {methods[idx]}:")
                print_eigens(eigenvals, eigenvecs)
            
            print(f"Time to find eigenvalues and eigenvectors using {methods[idx]}: {(qr_algo_end - qr_algo_start):.4f} seconds")
        
        A3 = self.A.copy()
        power_method_start = time.time()
        dom_eigen_value, dom_eigen_vector = power_method(A3, gb.args.test_maxiter, gb.args.test_tol)
        dom_eigen_vector = dom_eigen_vector.reshape((dom_eigen_vector.shape[0], 1))
        power_method_end = time.time()
        if len(dom_eigen_value) < 10 and len(dom_eigen_vector) < 10:
            print("Dominant eigenvector and eigenvalue using Power Method:")
            print_eigens(dom_eigen_value, dom_eigen_vector)
            
        print(f"Time to find eigenvalues and eigenvectors using Power Method: {(power_method_end - power_method_start):.4f} seconds")
