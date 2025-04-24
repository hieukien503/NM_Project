# This module implements the QR algorithm for computing the eigenvalues and eigenvectors of a matrix.
# It includes functions for Gram-Schmidt orthogonalization, Modified Gram-Schmidt orthogonalization,
# Householder reflections, Givens Rotations, and the QR algorithm itself.
# The QR algorithm is an iterative method for finding the eigenvalues and eigenvectors of a matrix.
# It is based on the QR decomposition of a matrix, which expresses the matrix as the product of an orthogonal matrix Q 
# and an upper triangular matrix R.
# The QR algorithm iteratively computes the QR decomposition of a matrix and updates the matrix by multiplying R and Q.
# The process continues until the matrix converges to a diagonal form, from which the eigenvalues can be easily extracted.

import numpy as np
import global_constant as gb

def project(u, v):
    """Project vector v onto vector u."""
    return (np.dot(u, v) / np.dot(u, u)) * u

def handle_special_case(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Check if A is a zero matrix
    if np.all(A == 0):
        return np.eye(A.shape[1]), np.zeros_like(A)
    
    # Check if A is an identity matrix
    if np.array_equal(A, np.eye(A.shape[1])):
        return np.eye(A.shape[1]), np.eye(A.shape[1])
    
    # Check if A is an orthogonal matrix
    try:
        if np.array_equal(A.T, np.linalg.inv(A)):
            return A.copy(), np.eye(A.shape[1])
    
    except np.linalg.LinAlgError as LAE:
        pass
    
    # Check if A is an upper triangular matrix
    def is_upper(A):
        for i in range(1, A.shape[0]):
            for j in range(0, i):
                if A[i, j] != 0:
                    return False
        
        return True
    
    if is_upper(A):
        return np.eye(A.shape[1]), A.copy()
    
    def is_diagonal(A):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i != j and A[i, j] != 0:
                    return False
                
        return True
    
    if is_diagonal(A):
        Q, R = np.eye(A.shape[1]), A.copy()
        for i in range(A.shape[1]):
            Q[i, i] = np.sign(A[i, i]) if np.sign(A[i, i]) != 0 else Q[i, i]
            R[i, i] = np.abs(A[i, i])
        
        return Q, R
    
    return None, None

def gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform Gram-Schmidt orthogonalization on the columns of matrix A."""

    n = A.shape[1]
    Q, R = handle_special_case(A)
    if Q is not None:
        return Q, R
    
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    U = np.zeros((n, n))
    for j in range(n):
        # Orthogonalize the j-th column of A against the previous columns
        # In Gram-Schmidt, we create a new orthogonal basis
        # by subtracting the projections of the previous columns from the current column
        U[:, j] = A[:, j] if j == 0 else \
                  A[:, j] - np.sum([project(U[:, i], A[:, j]) for i in range(j)], axis=0)
        
        Q[:, j] = U[:, j] / np.linalg.norm(U[:, j])
    
    # Compute the R matrix
    R = Q.T @ A
    return Q, R

def modified_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform Modified Gram-Schmidt orthogonalization on the columns of matrix A."""
    
    n = A.shape[1]
    Q, R = handle_special_case(A)
    if Q is not None:
        return Q, R
    
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    A_modified = A.copy()
    for i in range(n):
        # Orthogonalize the i-th column of A against the previous columns
        # In Modified Gram-Schmidt, we update the columns of A directly
        Q[:, i] = A_modified[:, i] / np.linalg.norm(A_modified[:, i])
        for j in range(i + 1, n):
            A_modified[:, j] -= project(Q[:, i], A_modified[:, j])
    
    # Compute the R matrix
    R = Q.T @ A
    return Q, R


def householder_reflections(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Householder reflection matrix and the vector."""

    n = A.shape[1]
    Q, R = handle_special_case(A)
    if Q is not None:
        return Q, R
    
    Q = np.eye(n)
    A_copy = A.copy()
    for k in range(n - 1):
        x = A[k:, k]                            # Extract the k-th column from the k-th row to the end
        e = np.zeros_like(x)                    # Create a zero vector of the same shape as x
        e[0] = np.linalg.norm(x)                # Set the first element of e to the norm of x
        u = x + np.sign(x[0]) * e               # Create the Householder vector
        v = u / np.linalg.norm(u)               # Normalize the Householder vector
        H = np.eye(n - k) - 2 * np.outer(v, v)  # Create the Householder matrix
        H = np.block([
            [np.eye(k), np.zeros((k, n - k))],  # Top-left block is the identity matrix
            [np.zeros((n - k, k)), H]           # Bottom-right block is the Householder matrix
        ])
        A_copy = H @ A_copy                     # Apply the Householder transformation to R
        Q = Q @ H.T                             # Update Q with the Householder transformation
    
    return Q, A_copy

def givens_rotations(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Givens rotation matrix and the vector."""

    n = A.shape[1]
    Q, R = handle_special_case(A)
    if Q is not None:
        return Q, R
    
    Q = np.eye(n)
    R = A.copy()
    for k in range(n - 1):
        for j in range(k + 1, n):
            angle = np.arctan2(-R[j, k], R[k, k])       # Compute the angle for the Givens rotation
            c = np.cos(angle)                           # Compute the cosine of the angle
            s = np.sin(angle)                           # Compute the sine of the angle
            G = np.eye(n)                               # Create an identity matrix of size n
            G[k, k] = G[j, j] = c                       # Set G[k, k] = G[j, j] = cos(angle)
            G[k, j] = -s                                # Set G[k, j] = -sin(angle)
            G[j, k] = s                                 # Set G[j, k] = sin(angle)
            R = G @ R                                   # Apply the Givens rotation to R
            Q = Q @ G.T                                 # Update Q with the Givens rotation
    
    return Q, R

def hessenberg(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def hr(x: np.ndarray) -> tuple[np.ndarray, float]:
        v = x.copy()
        v[0] += np.sign(x[0]) * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
        beta = 2.0
        return v, beta
    
    A = A.copy().astype(float)
    n = A.shape[0]
    Q_total = np.eye(n)

    for k in range(n - 2):
        x = A[(k + 1):, k]
        if np.allclose(x, 0):  # already zero below subdiagonal
            continue

        v, beta = hr(x)

        # Apply from the left: A[(k + 1):, k:] = (I - beta vvᵀ) A[(k + 1):, k:]
        A[(k + 1):, k:] -= beta * np.outer(v, v @ A[(k + 1):, k:])

        # Apply from the right: A[:, (k + 1):] = A[:, (k + 1):] (I - beta vvᵀ)
        A[:, (k + 1):] -= beta * np.outer(A[:, (k + 1):] @ v, v)

        # Accumulate Q (optional, if you want the orthogonal similarity)
        Q_k = np.eye(n)
        Q_k[(k + 1):, k+1:] -= beta * np.outer(v, v)
        Q_total = Q_total @ Q_k

    return A, Q_total  # A is now upper Hessenberg

def qr_algorithm(
        A: np.ndarray,
        method: str = 'cgs',
        tol: float = 1e-10,
        max_iter: int = 1000
    ) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenvalues and eigenvectors of matrix A using the QR algorithm."""
    
    Ak, Q_total = hessenberg(A)
    gb.matrices = [A]
    
    for _ in range(max_iter):
        if method == 'cgs':
            Q, R = gram_schmidt(Ak)

        elif method == 'mgs':
            Q, R = modified_gram_schmidt(Ak)

        elif method == 'hr':
            Q, R = householder_reflections(Ak)

        elif method == 'givens':
            Q, R = givens_rotations(Ak)

        else:
            raise ValueError("Invalid method specified")
        
        Ak_next = R @ Q
        Q_total = Q_total @ Q
        gb.matrices.append(Ak_next)
        
        # Check for convergence
        if np.linalg.norm(Ak - Ak_next, ord=gb.norm_ord) < tol:
            Ak = Ak_next
            break

        Ak = Ak_next
    
    eigenvalues = np.diag(Ak)
    return eigenvalues, Q_total