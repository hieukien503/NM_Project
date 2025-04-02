import argparse
import global_constant as gb

from utils import *
from test import TestCase

parser = argparse.ArgumentParser(
    description="Demonstrate QR Algorithm"
)
parser.add_argument("--gen", action="store_true", 
                    help="Generate a square matrix (use --sym for a symmetric matrix) and store it in the input file")
parser.add_argument("--sym", action="store_true", 
                    help="Enable symmetric matrix mode")
parser.add_argument("--complex", action="store_true",
                    help="Enable complex matrix mode")
parser.add_argument("--run", required=True, action="store_true",
                    help="Test the QR decomposition and algorithm.")
parser.add_argument("--maxsize", type=int, default=5,
                    help="Specify the maximum size of the generated matrix (only works if --gen or --test is enabled)")
parser.add_argument("--low", type=int, default=-200,
            help="Specify the lower bound for the elements of the generated matrix (only works if --gen or --test is enabled)")
parser.add_argument("--high", type=int, default=200,
            help="Specify the upper bound for the elements of the generated matrix (only works if --gen or --test is enabled)")
parser.add_argument("--eigens", choices=["cgs", "mgs", "hr", "givens", "power"], 
                    help="Run the QR algorithm using the specified method.")
parser.add_argument("--qr_decompo", choices=["cgs", "mgs", "chr", "mhr", "givens"], 
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

    if gb.args.gen:
        gen_sym_matrix(gb.args.input, gb.args.low, gb.args.high, gb.args.maxsize) if gb.args.sym else \
        gen_matrix(gb.args.input, gb.args.low, gb.args.high, gb.args.maxsize)
    
    if gb.args.visualize:
        gb.VISUALIZE = True

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