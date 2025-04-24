from sympy import *
from scipy.linalg import null_space
from utils import load_matrix, print_eigens, print_matrix, plot_power_method_convergence, plot_QR_algorithm_convergence
from qr_algorithm import *

import numpy as np
import global_constant as gb
import time

def characteristics_method(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenvalues and eigenvectors of matrix A using the characteristic polynomial method."""
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    
    n = A.shape[1]
    A_sym = Matrix([[Rational(item) for item in sublist] for sublist in A])
    lam = symbols('lambda')
    I = Matrix.eye(A_sym.shape[0])
    char_matrix = A_sym - lam * I
    char_poly = det(char_matrix).simplify()                 # Setup the characteristics polynomial det(A - lambda.I)
    solutions = solve(char_poly, lam)                       # Solve for lambda
    solutions = [re(val).evalf() for val in solutions if abs(val.as_real_imag()[1]) < gb.imag_tol]
    eigenvalues = []
    eigenvectors = []
    for eigenvalue in solutions:
        eig_eq = A - float(eigenvalue) * np.eye(n)
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
    if A.size == 0:
        raise ValueError("Matrix A must not be empty")
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    
    n = A.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    eigenvalue = 0.0
    
    for _ in range(max_iter):
        x_new = A @ x
        x_new /= np.linalg.norm(x_new)

        eigenvalue_new = (x_new.T @ A @ x_new) / (x_new.T @ x_new)
        difference = np.linalg.norm(x_new - x)

        if gb.VISUALIZE:
            gb.vectors.append(x_new)
            gb.eigenvalues.append(eigenvalue_new)
            
        if difference < tol:
            break
        
        x = x_new
        eigenvalue = eigenvalue_new
        
    return eigenvalue, x

class TestCase:
    def __init__ (self, filename: str):
        self.method = {
            "cgs": gram_schmidt,
            "mgs": modified_gram_schmidt,
            "hr": householder_reflections,
            "givens": givens_rotations
        }

        self.A = load_matrix(filename)

    def test_np_linalg_qr(self, method: str):
        methods = {
            "cgs": "Classical Gram-Schmidt Process",
            "mgs": "Modified Gram-Schmidt Process",
            "hr": "Householder Reflections",
            "givens": "Givens Rotations"
        }

        A1, A2 = self.A.copy(), self.A.copy()
        lib_time_start = time.time()
        Q_res, R_res = QR_decompostion(A1)
        lib_time_end = time.time()
        func_time_start = time.time()
        Q_test, R_test = gram_schmidt(A2)
        func_time_end = time.time()
        if self.A.shape[0] < 10:
            print("Matrix A:")
            print_matrix(self.A)
        
        print(f"Time to execute QR decomposition using numpy libraries: {(lib_time_end - lib_time_start):.4f} seconds")
        print(f"Time to execute QR decomposition using {methods[method]}: {(func_time_end - func_time_start):.4f} seconds")
        if Q_res.shape[1] < 10:
            print("Matrix Q (from numpy libraries):")
            print_matrix(Q_res)
        
        if R_res.shape[1] < 10:
            print("Matrix R (from numpy libraries):")
            print_matrix(R_res)
        
        if Q_test.shape[1] < 10:
            print(f"Matrix Q (from {methods[method]}):")
            print_matrix(Q_test)
        
        if R_test.shape[1] < 10:
            print(f"Matrix R (from {methods[method]}):")
            print_matrix(R_test)
    
    def test_eigen_01(self, method: str):
        methods = {
            "cgs": "Classical Gram-Schmidt Process",
            "mgs": "Modified Gram-Schmidt Process",
            "chr": "Classical Householder Reflections",
            "mhr": "Modified Householder Reflections",
            "givens": "Givens Rotations"
        }
        
        A1, A2 = self.A.copy(), self.A.copy()
        char_poly_start = time.time()
        eigenvalues, eigenvectors = characteristics_method(A1)
        char_poly_end = time.time()
        qr_algo_start = time.time()
        eigenvals, eigenvecs = qr_algorithm(A2, method, gb.args.test_tol, gb.args.test_maxiter)
        qr_algo_end = time.time()
        if self.A.shape[0] < 10:
            print("Matrix A:")
            print_matrix(self.A)
        
        if len(eigenvalues) < 10 and len(eigenvectors) < 10:
            print("Eigenvalues and Eigenvectors")
            print("Using CharPoly Method:")
            print_eigens(eigenvalues, eigenvectors)
        
        if len(eigenvals) < 10 and len(eigenvecs) < 10:
            print(f"Using QR Algorithm with {methods[method]}:")
            print_eigens(eigenvals, eigenvecs)
        
        print(f"Time to find eigenvalues and eigenvectors using CharPoly Method: {(char_poly_end - char_poly_start):.4f} seconds")
        print(f"Time to find eigenvalues and eigenvectors using QR Algorithm with {methods[method]}: {(qr_algo_end - qr_algo_start):.4f} seconds")
        if gb.VISUALIZE:
            plot_QR_algorithm_convergence(gb.matrices)
    
    def test_eigen_02(self, method: str):
        if method not in list(self.method.keys()): return
        methods = {
            "cgs": "Classical Gram-Schmidt Process",
            "mgs": "Modified Gram-Schmidt Process",
            "chr": "Classical Householder Reflections",
            "mhr": "Modified Householder Reflections",
            "givens": "Givens Rotations"
        }
        
        A1, A2 = self.A.copy(), self.A.copy()
        eigen_numpy_start = time.time()
        eigenvalues, eigenvectors = generate_eig_numpy(A1)
        eigen_numpy_end = time.time()
        qr_algo_start = time.time()
        eigenvals, eigenvecs = qr_algorithm(A2, method, gb.args.test_tol, gb.args.test_maxiter)
        qr_algo_end = time.time()
        if self.A.shape[0] < 10:
            print("Matrix A:")
            print_matrix(self.A)
        
        if len(eigenvalues) < 10 and len(eigenvectors) < 10:
            print("Eigenvalues and Eigenvectors")
            print("Using np.linalg.eig:")
            print_eigens(eigenvalues, eigenvectors)
        
        if len(eigenvals) < 10 and len(eigenvecs) < 10:
            print(f"Using QR Algorithm with {methods[method]}:")
            print_eigens(eigenvals, eigenvecs)

        print(f"Time to find eigenvalues and eigenvectors using np.linalg.eig: {(eigen_numpy_end - eigen_numpy_start):.4f} seconds")
        print(f"Time to find eigenvalues and eigenvectors using QR Algorithm with {methods[method]}: {(qr_algo_end - qr_algo_start):.4f} seconds")
        if gb.VISUALIZE:
            plot_QR_algorithm_convergence(gb.matrices)
    
    def test_eigen_03(self, method: str):
        if method not in list(self.method.keys()): return
        methods = {
            "cgs": "Classical Gram-Schmidt Process",
            "mgs": "Modified Gram-Schmidt Process",
            "chr": "Classical Householder Reflections",
            "mhr": "Modified Householder Reflections",
            "givens": "Givens Rotations"
        }
        
        A1, A2 = self.A.copy(), self.A.copy()
        eigen_sympy_start = time.time()
        eigenvalues, eigenvectors = generate_eig_sympy(A1)
        eigen_sympy_end = time.time()
        qr_algo_start = time.time()
        eigenvals, eigenvecs = qr_algorithm(A2, method, gb.args.test_tol, gb.args.test_maxiter)
        qr_algo_end = time.time()
        if self.A.shape[0] < 10:
            print("Matrix A:")
            print_matrix(self.A)
        
        if len(eigenvalues) < 10 and len(eigenvectors) < 10:
            print("Eigenvalues and Eigenvectors")
            print("Using sympy.Matrix.eigenvects:")
            print_eigens(eigenvalues, eigenvectors)
        
        if len(eigenvals) < 10 and len(eigenvecs) < 10:
            print(f"Using QR Algorithm with {methods[method]}:")
            print_eigens(eigenvals, eigenvecs)

        print(f"Time to find eigenvalues and eigenvectors using sympy.Matrix.eigenvects: {(eigen_sympy_end - eigen_sympy_start):.4f} seconds")
        print(f"Time to find eigenvalues and eigenvectors using QR Algorithm with {methods[method]}: {(qr_algo_end - qr_algo_start):.4f} seconds")
        if gb.VISUALIZE:
            plot_QR_algorithm_convergence(gb.matrices)

    def test_power_method(self, method: str):
        if method not in list(self.method.keys()): return
        methods = {
            "cgs": "Classical Gram-Schmidt Process",
            "mgs": "Modified Gram-Schmidt Process",
            "chr": "Classical Householder Reflections",
            "mhr": "Modified Householder Reflections",
            "givens": "Givens Rotations"
        }
        
        A1, A2 = self.A.copy(), self.A.copy()
        power_method_start = time.time()
        dom_eigen_value, dom_eigen_vector = power_method(A1, gb.args.test_maxiter, gb.args.test_tol)
        power_method_end = time.time()
        qr_algo_start = time.time()
        eigenvals, eigenvecs = qr_algorithm(A2, method, gb.args.test_tol, gb.args.test_maxiter)
        qr_algo_end = time.time()
        if self.A.shape[0] < 10:
            print("Matrix A:")
            print_matrix(self.A)
        
        if len(dom_eigen_vector) < 10:
            print("Dominant eigenvalue and eigenvector")
            print("Using Power Method:")
            print_eigens(dom_eigen_value, dom_eigen_vector)
        
        print(f"Using QR Algorithm with {methods[method]}:")
        idx = np.argmax(np.abs(eigenvals))
        dom_eigen_val = eigenvals[idx]
        dom_eigen_vec = eigenvecs[:, idx]
        if len(dom_eigen_vec) < 10:
            print_eigens(dom_eigen_val, dom_eigen_vec)
        
        print(f"Time to find eigenvalues and eigenvectors using Power Method: {(power_method_end - power_method_start):.4f} seconds")
        print(f"Time to find eigenvalues and eigenvectors using QR Algorithm with {methods[method]}: {(qr_algo_end - qr_algo_start):.4f} seconds")
        if gb.VISUALIZE:
            plot_power_method_convergence(np.array(gb.vectors), np.array(gb.eigenvalues))