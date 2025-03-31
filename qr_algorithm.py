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

def gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform Gram-Schmidt orthogonalization on the columns of matrix A."""

    n = A.shape[1]
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

def householder_reflection(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Householder reflection matrix and the vector."""

    n = A.shape[1]
    Q = np.eye(n)
    R = A.copy()
    for k in range(n - 1):
        x = R[k:, k]                            # Extract the k-th column from the k-th row to the end
        e = np.zeros_like(x)                    # Create a zero vector of the same shape as x
        e[0] = np.linalg.norm(x)                # Set the first element of e to the norm of x
        u = x - e                               # Create the Householder vector
        v = u / np.linalg.norm(u)               # Normalize the Householder vector
        H = np.eye(n - k) - 2 * np.outer(v, v)  # Create the Householder matrix
        H = np.block([
            [np.eye(k), np.zeros((k, n - k))],  # Top-left block is the identity matrix
            [np.zeros((n - k, k)), H]           # Bottom-right block is the Householder matrix
        ])
        R = H @ R                               # Apply the Householder transformation to R
        Q = Q @ H.T                             # Update Q with the Householder transformation
    
    return Q, R

def modified_householder_reflection(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Modified Householder reflection matrix and the vector."""

    n = A.shape[1]
    Q = np.eye(n)
    R = A.copy()
    for k in range(n - 1):
        x = R[k:, k]                            # Extract the k-th column from the k-th row to the end
        e = np.zeros_like(x)                    # Create a zero vector of the same shape as x
        e[0] = np.linalg.norm(x)                # Set the first element of e to the norm of x
        u = x + np.sign(x[0]) * e               # Create the Householder vector (key difference is here)
        v = u / np.linalg.norm(u)               # Normalize the Householder vector
        H = np.eye(n - k) - 2 * np.outer(v, v)  # Create the Householder matrix
        H = np.block([
            [np.eye(k), np.zeros((k, n - k))],  # Top-left block is the identity matrix
            [np.zeros((n - k, k)), H]           # Bottom-right block is the Householder matrix
        ])
        R = H @ R                               # Apply the Householder transformation to R
        Q = Q @ H.T                             # Update Q with the Householder transformation
    
    return Q, R

def givens_rotations(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Givens rotation matrix and the vector."""

    n = A.shape[1]
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

def qr_algorithm(
        A: np.ndarray,
        method: str = 'gram_schmidt',
        tol: float = 1e-10,
        max_iter: int = 1000
    ) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenvalues and eigenvectors of matrix A using the QR algorithm."""

    # Check if the matrix is empty
    if A.size == 0:
        raise ValueError("Matrix A must not be empty")
    
    # Check if the matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    
    n = A.shape[0]
    Q_total = np.eye(n)
    Ak = A.copy()
    gb.matrices = [A]
    
    for i in range(max_iter):
        if method == 'cgs':
            Q, R = gram_schmidt(Ak)

        elif method == 'mgs':
            Q, R = modified_gram_schmidt(Ak)

        elif method == 'chr':
            Q, R = householder_reflection(Ak)

        elif method == 'mhr':
            Q, R = modified_householder_reflection(Ak)

        elif method == 'givens':
            Q, R = givens_rotations(Ak)

        else:
            raise ValueError("Invalid method specified")
        
        Ak_next = R @ Q
        Q_total = Q_total @ Q
        gb.matrices.append(Ak_next)
        
        # Check for convergence
        if np.allclose(Ak, Ak_next, atol=tol):
            Ak = Ak_next
            break

        Ak = Ak_next
    
    eigenvalues = np.diag(Ak)
    eigenvectors = Q_total
    return eigenvalues, eigenvectors