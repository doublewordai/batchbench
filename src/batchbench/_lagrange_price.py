"""Weighted least squares solver used by batchbench pricing utilities."""

from __future__ import annotations

import numpy as np
import cvxpy as cp

def solve_wls_weighted_profit(A, b, w=None, eps=0.0):
    """Solve the constrained weighted least-squares problem described in docs.

    Parameters mirror scripts/lagrange_price.py for compatibility.  The solver is
    intentionally kept dependency-light and exposed as part of the Python
    package so it can be reused by the CLI and library users alike.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape

    if w is None:
        w = np.ones(m, dtype=float)
    else:
        w = np.asarray(w, dtype=float).ravel()
        if w.shape != (m,):
            raise ValueError("w must have shape (m,)")

    if profit_weights is None:
        u = w
    else:
        u = np.asarray(profit_weights, dtype=float).ravel()
        if u.shape != (m,):
            raise ValueError("profit_weights must have shape (m,)")

    Aw = A * w[:, None]
    H = A.T @ Aw

    g = A.T @ (w * b)
    c = A.T @ u
    d = (1.0 + eps) * float(u @ b)

    def solve_H(rhs):
        try:
            return np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            H_pinv = np.linalg.pinv(H)
            return H_pinv @ rhs

    x = solve_H(g)

    income = float(c @ x)
    cost = d
    if income >= cost:
        print("Unconstrained solution satisfies constraint; returning it directly.")
        
    else:
        print("Adjusted solution to satisfy constraint.")
        Hinvc = solve_H(c)
        denom = float(c @ Hinvc)

        if denom <= 0:
            raise RuntimeError(
                "Degenerate constraint direction: c^T H^{-1} c <= 0. Check data or add ridge."
            )

        tau = (d - float(c @ x)) / denom
        x = x + tau * Hinvc        
    

    income = float(c @ x)
    cost = d
    print(f"Income is {income}, cost is {cost}.")

    for i, row in enumerate(A):
        row_income = np.dot(row, x)
        print(f"Row {i} income is {row_income}.")
        print(f"Row {i} cost is {b[i]}.")
    return x*1_000_000


def solve_wls_weighted_profit_nonneg(A, b, w=None, eps=0.0):
    """
    Weighted least squares with:
      - weighted fit:    minimize (1/2) (Ax - b)^T W (Ax - b)
      - weighted profit: (A^T u)^T x >= (1+eps) * (u^T b)
      - non-negativity:  x >= min_price

    Parameters
    ----------
    A : array_like, shape (m, n)
        Row i: features (e.g. [input_tokens_i, output_tokens_i]).
    b : array_like, shape (m,)
        GPU costs per request.
    w : array_like, shape (m,), optional
        Row weights for the least-squares fit. Defaults to all ones.
    eps : float, optional
        Profit margin buffer: enforce revenue >= (1+eps) * weighted cost.

    Returns
    -------
    x : ndarray, shape (n,)
        Optimal prices.
    """

    # Ensure arrays and shapes
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape

    # Fit weights
    if w is None:
        w = np.ones(m, dtype=float)
    else:
        w = np.asarray(w, dtype=float).ravel()
        if w.shape != (m,):
            raise ValueError("w must have shape (m,)")

    # Weighted LS: use sqrt(w) in the objective
    Wsqrt = np.sqrt(w)

    # Profitability constraint: (A^T u)^T x >= (1+eps) * (u^T b)
    c = A.T @ w                                # shape (n,)
    d = (1.0 + eps) * float(w @ b)             # scalar

    # Define variable
    x = cp.Variable(n)

    # Objective: sum_i w_i (a_i^T x - b_i)^2
    residual = A @ x - b
    obj = cp.Minimize(cp.norm1(cp.multiply(Wsqrt, residual)))

    # Constraints
    constraints = [
        c @ x >= d,           # weighted profitability
        x[0] >= 1e-8,         # non-negativity (or >= small positive)
    ]

    prob = cp.Problem(obj, constraints)
    # OSQP or ECOS are both fine; OSQP is often good for QPs with inequalities
    prob.solve(solver="OSQP", verbose=False)
        
    res = np.array(x.value).ravel() * 1_000_000
    return res[0], res[1]


__all__ = ["solve_wls_weighted_profit"]
