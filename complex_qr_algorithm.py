# Complex version of QR decomposition and QR algorithm

import numpy as np
import global_constant as gb

def gram_schmidt_complex(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[1]
    Q = np.zeros(shape=(n, n), dtype=complex)
    R = Q.copy()
    A_copy = A.copy()
    for j in range(n):
        v = A_copy[:, j]
        for i in range(j):          
            R[i, j] = np.vdot(Q[:, j], v)                   # vdot(u, v) = conj(u).v = <u, v>
            v = v - R[i, j] * Q[:, i]                       # w_k = v_k - sum_{j = 1}^{k - 1} <v_k, u_j>u_j
        
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R

def modified_gram_schmidt_complex(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[1]
    Q = np.zeros(shape=(n, n), dtype=complex)
    R = Q.copy()
    V = A.copy()
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        for i in range(j + 1, n):
            R[j, i] = np.vdot(Q[:, j], V[:, i])
            V[:, i] = V[:, i] - R[j, i] * Q[:, j]

    return Q, R

def householder_reflections_complex(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[1]
    R = A.copy().astype(complex)
    Q = np.eye(n, dtype=complex)
    for k in range(n):
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x)
        u = x - e1
        u = u / np.linalg.norm(u)
        H = np.eye(n, dtype=complex)
        H[k:, k:] -= 2.0 * np.outer(u, np.conj(u))
        R = H @ R
        Q = Q @ H.T.conj()

    return Q, R

def givens_rotations_complex(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def evaluate_cs(a, b):
        if b == 0:
            return 1, 0
        
        r = np.hypot(a, b)
        c = a / r
        s = -b / r
        return c, s
    
    n = A.shape[1]
    Q = np.eye(n, dtype=complex)
    R = A.copy().astype(complex)
    for j in range(n):
        for i in range(n - 1, j, -1):
            c, s = evaluate_cs(R[i - 1, j], R[i, j])
            G = np.eye(n, dtype=complex)
            G[[i-1, i], [i-1, i]] = c
            G[i, i-1] = s
            G[i-1, i] = -s
            R = G @ R
            Q = Q @ G.T.conj()

    return Q, R

def qr_algorithm_complex(
        A: np.ndarray,
        method: str = "cgs",
        max_iter: int = 1000,
        tol: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:

    n = A.shape[1]
    Ak = A.copy().astype(complex)
    Q_total = np.eye(n, dtype=complex)
    for _ in range(max_iter):
        if method == 'cgs':
            Q, R = gram_schmidt_complex(Ak)

        elif method == 'mgs':
            Q, R = modified_gram_schmidt_complex(Ak)

        elif method == 'hr':
            Q, R = householder_reflections_complex(Ak)

        elif method == 'givens':
            Q, R = givens_rotations_complex(Ak)

        else:
            raise ValueError("Invalid method specified")
        
        Ak_new = R @ Q
        Q_total = Q_total @ Q
        gb.matrices.append(Ak_new)

        if np.linalg.norm(Ak - Ak_new, ord=gb.norm_ord) < tol:
            Ak = Ak_new
            break

        Ak = Ak_new
    
    eigenvalues = np.diag(Ak)
    eigenvectors = Q_total
    return eigenvalues, eigenvectors