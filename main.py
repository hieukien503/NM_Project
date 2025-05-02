import argparse
import global_constant as gb
import time

from utils import *
from test import TestCase
from qr_algorithm import *

parser = argparse.ArgumentParser(
    description="Demonstrate QR Algorithm"
)
parser.add_argument("--gen", action="store_true", 
                    help="Generate a square matrix (use --sym for a symmetric matrix) and store it in the input file")
parser.add_argument("--sym", action="store_true", 
                    help="Enable symmetric matrix mode")
parser.add_argument("--run", choices=["qr", "wilkinson", "francis"], default=None,
                    help="Run the QR algorithm with the specified method.")
parser.add_argument("--test", action="store_true",
                    help="Test the QR algorithm.")
parser.add_argument("--maxsize", type=int, default=5,
                    help="Specify the maximum size of the generated matrix (only works if --gen is enabled)")
parser.add_argument("--low", type=int, default=-200,
            help="Specify the lower bound for the elements of the generated matrix (only works if --gen is enabled)")
parser.add_argument("--high", type=int, default=200,
            help="Specify the upper bound for the elements of the generated matrix (only works if --gen is enabled)")
parser.add_argument("--input", type=str, required=True,
                    help="Input file contains the matrix.")
parser.add_argument("--visualize", action="store_true", 
                    help="Visualize the convergence")
parser.add_argument("--maxiter", type=int, default=1000,
                    help="Maximum iteration")
parser.add_argument("--tolerance", type=float, default=1e-6, 
                    help="Convergence tolerance")
parser.add_argument("--test_maxiter", type=int, default=1000, 
                    help="Testing maximum iteration")
parser.add_argument("--test_tol", type=float, default=1e-6, 
                    help="Testing tolerance")

gb.args = parser.parse_args()
if __name__ == '__main__':
    if not gb.args.run and not gb.args.test:
        raise ValueError("Must specify --run or --test")
    
    if gb.args.run and gb.args.test:
        raise ValueError("Only --run or --test is specified during the project")

    if gb.args.gen:
        gen_sym_matrix(gb.args.input, gb.args.low, gb.args.high, gb.args.maxsize) if gb.args.sym else \
        gen_matrix(gb.args.input, gb.args.low, gb.args.high, gb.args.maxsize)
    
    if gb.args.visualize:
        gb.VISUALIZE = True

    if gb.args.test:
        testcase = TestCase(filename=gb.args.input)
        testcase.test_eigen()
    
    else:
        methods = {
            "qr": qr_algorithm,
            "wilkinson": qr_algorithm_wilkinson,
            "francis": francis_double_shift_qr
        }

        method_names = {
            "qr": "QR Algorithm",
            "wilkinson": "QR Algorithm with Wilkinson Shift",
            "francis": "Francis Double Shift QR"
        }

        if gb.args.run not in methods:
            raise ValueError(f"Invalid method: {gb.args.run}. Choose from {list(methods.keys())}.")
        
        method = methods[gb.args.run]
        A = load_matrix(gb.args.input)
        if A.shape[0] < 10:
            print("Matrix A:")
            print_matrix(A)

        A2 = A.copy()
        qr_algo_start = time.time()
        eigenvals, eigenvecs = method(A2, gb.args.test_maxiter, gb.args.test_tol)
        qr_algo_end = time.time()
        
        if len(eigenvals) < 10 and len(eigenvecs) < 10:
            print(f"Using QR Algorithm with {method}:")
            print_eigens(eigenvals, eigenvecs)
        
        print(f"Time to find eigenvalues and eigenvectors using {method_names[gb.args.run]}: {(qr_algo_end - qr_algo_start):.4f} seconds")
        if gb.VISUALIZE:
            plot_QR_algorithm_convergence(gb.matrices)