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
parser.add_argument("--run", action="store_true",
                    help="Run the QR decomposition and algorithm.")
parser.add_argument("--test", action="store_true",
                    help="Test the QR decomposition and algorithm.")
parser.add_argument("--maxsize", type=int, default=5,
                    help="Specify the maximum size of the generated matrix (only works if --gen or --test is enabled)")
parser.add_argument("--low", type=int, default=-200,
            help="Specify the lower bound for the elements of the generated matrix (only works if --gen or --test is enabled)")
parser.add_argument("--high", type=int, default=200,
            help="Specify the upper bound for the elements of the generated matrix (only works if --gen or --test is enabled)")
parser.add_argument("--eigens", choices=["cgs", "mgs", "hr", "givens", "power"], 
                    help="Run the QR algorithm using the specified method.")
parser.add_argument("--qr_decompo", choices=["cgs", "mgs", "hr", "givens"], 
                    help="Perform QR decomposition using the specified method.")
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
    if gb.args.eigens and gb.args.qr_decompo:
        raise ValueError("Only --eigens or --qr_decompo is specified during the project")
    
    if not gb.args.eigens and not gb.args.qr_decompo:
        raise ValueError("Must specify --eigens or --qr_decompo")
    
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
        if gb.args.qr_decompo:
            testcase.test_np_linalg_qr(gb.args.qr_decompo)
        
        else:
            if gb.args.eigens != "power":
                while True:
                    option = input("Choose testcase (1: CharPoly, 2: numpy.linalg.eig, 3: sympy.Matrix.eigenvects): ")
                    match option:
                        case "1":
                            testcase.test_eigen_01(gb.args.eigens)
                            break

                        case "2":
                            testcase.test_eigen_02(gb.args.eigens)
                            break

                        case "3":
                            testcase.test_eigen_03(gb.args.eigens)
                            break

                        case _:
                            continue
            
            else:
                testcase.test_power_method(gb.args.eigens)
    
    else:
        if gb.args.qr_decompo:
            method = gb.args.qr_decompo
            methods = {
                "cgs": "Gram-Schmidt Process",
                "mgs": "Modified Gram-Schmidt Process",
                "hr": "Householder Reflections",
                "givens": "Givens Rotations"
            }

            method_func = {
                "cgs": gram_schmidt,
                "mgs": modified_gram_schmidt,
                "hr": householder_reflections,
                "givens": givens_rotations
            }
            
            A = load_matrix(gb.args.input)
            A_copy = A.copy()
            func_time_start = time.time()
            Q, R = method_func[method](A_copy)
            func_time_end = time.time()
            if A.shape[1] < 10:
                print("Matrix A:")
                print_matrix(A)
            
            print(f"Time to execute QR decomposition using {methods[method]}: {(func_time_end - func_time_start):.4f} seconds")
            if Q.shape[1] < 10:
                print(f"Matrix Q (from {methods[method]}):")
                print_matrix(Q)
            
            if R.shape[1] < 10:
                print(f"Matrix R (from {methods[method]}):")
                print_matrix(R)
        
        else:
            method = gb.args.eigens
            methods = {
                "cgs": "Gram-Schmidt Process",
                "mgs": "Modified Gram-Schmidt Process",
                "hr": "Householder Reflections",
                "givens": "Givens Rotations"
            }
            A = load_matrix(gb.args.input)
            A_copy = A.copy()
            qr_time_start = time.time()
            eigenvals, eigenvecs = qr_algorithm(A_copy, method, gb.args.tolerance, gb.args.maxiter)
            qr_time_end = time.time()
            if A.shape[0] < 10:
                print("Matrix A:")
                print_matrix(A)
            
            if len(eigenvals) < 10 and len(eigenvecs) < 10:
                print(f"Using QR Algorithm with {methods[method]}:")
                print_eigens(eigenvals, eigenvecs)
            
            print(f"Time to find eigenvalues and eigenvectors using QR Algorithm with {methods[method]}: {(qr_time_end - qr_time_start):.4f} seconds")
            if gb.VISUALIZE:
                plot_QR_algorithm_convergence(gb.matrices)