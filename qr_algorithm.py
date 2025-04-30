import numpy as np
import global_constant as gb
from scipy.linalg import hessenberg

def qr_algorithm(
    A: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenvalues and eigenvectors of matrix A using the QR algorithm."""
    
    Ak = hessenberg(A)
    Q_total = np.eye(A.shape[0])
    gb.matrices = [A]
    
    for _ in range(max_iter):
        Q, R = np.linalg.qr(Ak, mode='complete')
        
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

def extract_eigens_from_schur(
    T: np.ndarray,
    Q_total: np.ndarray,
    tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Extract eigenvalues and eigenvectors from the Schur form."""
    n = T.shape[0]
    eigenvalues = []
    eigenvectors = []
    i = 0
    while i < n:
        if i < n - 1 and abs(T[i + 1, i]) > tol:
            # 2×2 block → complex conjugate eigenvalues
            H = T[i : (i + 2), i : (i + 2)]
            # Fast eigenvalue computation for 2x2 matrix
            eigvals, eigvects = np.linalg.eig(H)
            for j in range(2):
                w = eigvects[:, j]
                v = Q_total[:, i : (i + 2)] @ w
                eigenvalues.append(eigvals[j])
                eigenvectors.append(v)
            
            i += 2

        else:
            eigvals = T[i, i]
            e = np.zeros(n)
            e[i] = 1.0
            v = Q_total @ e
            eigenvectors.append(v)
            eigenvalues.append(T[i, i])
            i += 1

    return np.array(eigenvalues), np.column_stack(eigenvectors)

def qr_algorithm_wilkinson(
    A: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the eigenvalues of a real square matrix A using the QR algorithm
    with the Wilkinson shift strategy.

    Args:
        A (np.ndarray): The real square matrix whose eigenvalues are to be found.
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for convergence.

    Returns:
        tuple[np.ndarray, np.ndarray]: An tuple contains the eigenvalues and the
        eigenvectors of the matrix A.
    """
    def wilkinson_shift(H):
        """Compute the Wilkinson shift for the given matrix H."""
        n = H.shape[0]
        if n < 2:
            return H[0, 0] if n == 1 else 0

        a = H[n - 2, n - 2]
        b = H[n - 2, n - 1]
        c = H[n - 1, n - 2]
        d = H[n - 1, n - 1]

        roots = np.roots([1, -(a + d), a * d - b * c])
        # Choose the root closest to the bottom-right eigenvalue
        mu = roots[np.argmin(np.abs(roots - d))]
        return mu
    
    n = A.shape[0]
    if n == 1:
        return A[0, 0]

    H = hessenberg(A)
    H = H.astype(complex)
    gb.matrices.append(H.copy())
    Q_total = np.eye(n, dtype=complex)
    iterations = 0
    m = n

    while m > 1:
        if iterations > max_iter:
            break

        # Wilkinson shift
        mu = wilkinson_shift(H[:m, :m])
        shift_matrix = mu * np.identity(m)
        Q, R = np.linalg.qr(H[:m, :m] - shift_matrix)
        H[:m, :m] = R @ Q + shift_matrix
        Q_full = np.eye(n, dtype=complex)
        Q_full[:m, :m] = Q
        Q_total = Q_total @ Q_full

        iterations += 1

        # Check for deflation (small off-diagonal element)
        if abs(H[m - 1, m - 2]) < tol * (abs(H[m - 2, m - 2]) + abs(H[m - 1, m - 1])):
            H[m - 1, m - 2] = 0
            m -= 1
        
        gb.matrices.append(H.copy())

    return extract_eigens_from_schur(H, Q_total, tol=tol)

def francis_double_shift_qr(
    H: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the eigenvalues of a real square matrix A using the QR algorithm
    with the Francis Double Shift strategy.

    Args:
        A (np.ndarray): The real square matrix whose eigenvalues are to be found.
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for convergence.

    Returns:
        tuple[np.ndarray, np.ndarray]: An tuple contains the eigenvalues and the
        eigenvectors of the matrix A.
    """
    def householder_reflector(x):
        """Compute Householder matrix P such that P @ x = alpha * e1"""
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            return np.eye(len(x), dtype=complex)
        
        v = x.copy()
        sign = x[0] / abs(x[0]) if x[0] != 0 else 1.0
        v[0] += sign * np.linalg.norm(x)
        v /= np.linalg.norm(v)
        P = np.eye(len(x)) - 2 * np.outer(v, v)
        return P

    def givens_rotation(a, b):
        """Compute Givens rotation matrix G such that G.T @ [a; b] = [r; 0]"""
        if a == 0 and b == 0:
            return np.eye(2, dtype=complex)
        
        r = np.linalg.norm([a, b])
        c, s = a / r, -b / r if r != 0 else (1, 0)
        return np.array([[c, -np.conj(s)], [s, np.conj(c)]], dtype=complex)

    H = hessenberg(H)
    H = H.astype(complex)
    n = H.shape[0]
    gb.matrices.append(H.copy())
    Q_total = np.eye(n, dtype=complex)
    p = n
    iter_count = 0

    while p > 2 and iter_count < max_iter:
        iter_count += 1
        q = p - 1

        # Create Wilkinson shift
        s = H[q - 1, q - 1] + H[p - 1, p - 1]
        t = H[q - 1, q - 1] * H[p - 1, p - 1] - H[q - 1, p - 1] * H[p - 1, q - 1]

        # Compute first column of M
        x = H[0, 0] ** 2 + H[0, 1] * H[1, 0] - s * H[0, 0] + t
        y = H[1, 0] * (H[0, 0] + H[1, 1] - s)
        z = H[1, 0] * H[2, 1]

        for k in range(p - 2):
            r = max(1, k)
            u = np.array([x, y, z])
            Pk = householder_reflector(u)

            # Apply from the left
            r_end = min(k + 4, p)
            H[k : (k + 3), (r - 1) : n] = Pk @ H[k : (k + 3), (r - 1) : n]

            # Apply from the right
            H[0 : r_end, k : (k + 3)] = H[0 : r_end, k : (k + 3)] @ Pk.T

            Pk_full = np.eye(n, dtype=complex)
            Pk_full[k : (k + 3), k : (k + 3)] = Pk
            Q_total = Q_total @ Pk_full.T

            x = H[k + 1, k]
            y = H[k + 2, k]
            if k < p - 3:
                z = H[k + 3, k]

        x = H[p - 2, p - 3]
        y = H[p - 1, p - 3]
        G = givens_rotation(x, y)

        # Apply from the left
        H[(q - 1) : p, (p - 3) : n] = G @ H[(q - 1) : p, (p - 3) : n]

        # Apply from the right
        H[0 : p, (p - 2) : p] = H[0 : p, (p - 2) : p] @ G.T
        Gfull = np.eye(n, dtype=complex)
        Gfull[(p - 2) : p, (p - 2) : p] = G
        Q_total = Q_total @ Gfull.T

        # Check for convergence
        if abs(H[p - 1, q - 1]) < tol * (abs(H[q - 1, q - 1]) + abs(H[p - 1, p - 1])):
            H[p - 1, q - 1] = 0
            gb.matrices.append(H.copy())
            p -= 1
            q = p - 1

        elif abs(H[p - 2, q - 2]) < tol * (abs(H[q - 2, q - 2]) + abs(H[q - 1, q - 1])):
            H[p - 2, q - 2] = 0
            gb.matrices.append(H.copy())
            p -= 2
            q = p - 1

        else:
            gb.matrices.append(H.copy())
            pass  # No convergence yet

    return extract_eigens_from_schur(H, Q_total, tol=tol)