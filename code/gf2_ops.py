#-- coding: utf-8 -*-
# gf2_ops.py

import torch
from typing import List, Optional, Union
import abc

# TorchScript helpers for performance-critical routines
@torch.jit.script
def _gf2_matmul_script(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    GF(2) matrix–vector / matrix–matrix product.

    • CPU  : use int32 -> fast, avoids fp round‑off
    • CUDA : fall back to float32 because integer matmul kernels are absent
    """
    if M.is_cuda:
        # float32 GEMM is available and fast on all GPUs
        tmp = torch.matmul(M.to(torch.float16), v.to(torch.float16))
    else:
        tmp = torch.matmul(M.to(torch.int32),  v.to(torch.int32))
    tmp = torch.bitwise_and(tmp.to(torch.int32), 1)
    return tmp.to(torch.int8)


@torch.jit.script
def _invert_matrix_script(M_in: torch.Tensor) -> torch.Tensor:
    n = M_in.size(0)
    A = M_in.to(torch.int8).clone()
    I = torch.eye(n, dtype=torch.int8, device=M_in.device)
    for col in range(n):
        pivot_row = col
        while pivot_row < n and A[pivot_row, col] != 1:
            pivot_row += 1
        if pivot_row == n:
            raise RuntimeError("Matrix is not invertible over GF(2)")
        if pivot_row != col:
            tmp = A[col].clone()
            A[col] = A[pivot_row]
            A[pivot_row] = tmp
            tmp = I[col].clone()
            I[col] = I[pivot_row]
            I[pivot_row] = tmp
        for row in range(n):
            if row != col and A[row, col] == 1:
                A[row] = torch.bitwise_xor(A[row], A[col])
                I[row] = torch.bitwise_xor(I[row], I[col])
    return I

class AbstractGF2Ops(abc.ABC):
    """Abstract Base Class for GF(2) operations."""
    
    @staticmethod
    @abc.abstractmethod
    def matmul(M: torch.Tensor, v: torch.Tensor, validate: bool = True) -> torch.Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def invert_matrix(M_in: torch.Tensor, validate: bool = True) -> torch.Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def add(A: torch.Tensor, B: torch.Tensor, validate: bool = True) -> torch.Tensor:
        pass


class GF2Ops(AbstractGF2Ops):
    """
    Simplified GPU-optimized GF(2) operations.
    
    Key optimizations:
    - Uses int8 for memory efficiency
    - Vectorized operations for GPU parallelism
    - In-place operations to minimize memory allocation
    - Efficient bitwise operations
    """
    
    _EPSILON = 1e-6
    _PREFERRED_DTYPE = torch.int8
    
    @staticmethod
    def _validate_tensor(tensor: torch.Tensor) -> None:
        """Validate that tensor contains only 0 and 1."""
        if tensor.dtype == torch.bool:
            return
        if tensor.dtype == torch.int8:
            if not ((tensor == 0) | (tensor == 1)).all():
                raise ValueError("Tensor must only contain 0 or 1")
            return
            
        if tensor.dtype in [torch.float32, torch.float64]:
            valid = ((tensor - 0.0).abs() < GF2Ops._EPSILON).logical_or(
                    (tensor - 1.0).abs() < GF2Ops._EPSILON).all()
            if not valid:
                raise ValueError("Tensor must only contain 0.0 or 1.0")
        else:
            valid = tensor.eq(0).logical_or(tensor.eq(1)).all()
            if not valid:
                raise ValueError("Tensor must only contain 0 or 1")
    
    @staticmethod
    def matmul(M: torch.Tensor, v: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """GPU-optimized GF(2) matrix multiplication."""
        if validate:
            GF2Ops._validate_tensor(M)
            GF2Ops._validate_tensor(v)
        
        # Convert to efficient dtype
        M_work = M.to(torch.int8) if M.dtype != torch.int8 else M
        v_work = v.to(torch.int8) if v.dtype != torch.int8 else v

        # TorchScripted kernel for efficiency
        result = _gf2_matmul_script(M_work, v_work)
        
        # Return in original dtype
        if M.dtype == torch.bool or v.dtype == torch.bool:
            return result.bool()
        return result.to(M.dtype)
    
    @staticmethod
    def add(A: torch.Tensor, B: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """GPU-optimized GF(2) addition using XOR."""
        if validate:
            GF2Ops._validate_tensor(A)
            GF2Ops._validate_tensor(B)
        
        # For boolean tensors, use logical_xor directly
        if A.dtype == torch.bool and B.dtype == torch.bool:
            return torch.logical_xor(A, B)
        
        # For other types, use bitwise XOR
        A_work = A.to(torch.int8) if A.dtype != torch.int8 else A
        B_work = B.to(torch.int8) if B.dtype != torch.int8 else B
        
        result = torch.bitwise_xor(A_work, B_work)
        
        # Return in original dtype
        if A.dtype == torch.bool:
            return result.bool()
        return result.to(A.dtype)
    
    @staticmethod
    def invert_matrix(M_in: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """GPU-optimized matrix inversion over GF(2) - CORRECTED VERSION."""
        if validate:
            GF2Ops._validate_tensor(M_in)
        
        # Handle batch dimensions
        if M_in.dim() > 2:
            return GF2Ops._batch_invert(M_in)
        
        if M_in.shape[0] != M_in.shape[1]:
            raise ValueError("Matrix must be square to invert")
        
        # Use TorchScripted kernel for efficiency and normalize errors
        try:
            result = _invert_matrix_script(M_in)
            if M_in.dtype == torch.bool:
                return result.bool()
            return result.to(M_in.dtype)
        except Exception as e:
            raise RuntimeError(str(e))


    @staticmethod
    def _batch_invert(matrices: torch.Tensor) -> torch.Tensor:
        """Batch matrix inversion over GF(2)."""
        shape = matrices.shape
        n = shape[-1]
        if shape[-2] != n:
            raise ValueError("All matrices must be square")
        
        # For small matrices, use vectorized approach
        if n <= 4:
            return GF2Ops._batch_small_matrix_inverse(matrices)
        
        # For larger matrices, process individually
        batch_shape = shape[:-2]
        batch_size = torch.prod(torch.tensor(batch_shape)).item()
        matrices_flat = matrices.reshape(batch_size, n, n)
        
        results = torch.zeros_like(matrices_flat, dtype=matrices.dtype)
        
        for i in range(batch_size):
            try:
                results[i] = GF2Ops.invert_matrix(matrices_flat[i], validate=False)
            except RuntimeError as e:
                idx = torch.unravel_index(i, batch_shape)
                raise RuntimeError(f"Matrix at index {idx} is not invertible: {str(e)}")
        
        return results.reshape(shape)
    
    @staticmethod
    def _batch_small_matrix_inverse(matrices: torch.Tensor) -> torch.Tensor:
        """Optimized batch inversion for small matrices."""
        shape = matrices.shape
        n = shape[-1]
        device = matrices.device
        
        # Flatten batch dimensions
        matrices_flat = matrices.reshape(-1, n, n)
        batch_size = matrices_flat.shape[0]
        
        if n == 1:
            # 1x1 matrices
            result = matrices_flat.clone()
            if (result == 0).any():
                raise RuntimeError("Found non-invertible 1x1 matrix")
            return result.reshape(shape)
        
        if n == 2:
            # 2x2 matrices - fully vectorized
            a = matrices_flat[:, 0, 0]
            b = matrices_flat[:, 0, 1]
            c = matrices_flat[:, 1, 0]
            d = matrices_flat[:, 1, 1]
            
            # Determinant in GF(2)
            det = (a * d + b * c) & 1
            
            if (det == 0).any():
                raise RuntimeError("Found non-invertible 2x2 matrix")
            
            # Inverse formula
            inv = torch.zeros_like(matrices_flat, dtype=matrices.dtype)
            inv[:, 0, 0] = d
            inv[:, 0, 1] = b
            inv[:, 1, 0] = c
            inv[:, 1, 1] = a
            
            return inv.reshape(shape).to(matrices.dtype)
        
        # For 3x3 and 4x4, fall back to individual processing
        results = torch.zeros_like(matrices_flat, dtype=matrices.dtype)
        for i in range(batch_size):
            results[i] = GF2Ops.invert_matrix(matrices_flat[i], validate=False)
        
        return results.reshape(shape)
    
    @staticmethod
    def rank(M: torch.Tensor, validate: bool = True) -> Union[int, torch.Tensor]:
        """Compute rank of matrix over GF(2)."""
        if validate:
            GF2Ops._validate_tensor(M)
        
        if M.dim() > 2:
            # Batched rank
            shape = M.shape
            batch_shape = shape[:-2]
            batch_size = torch.prod(torch.tensor(batch_shape)).item()
            M_flat = M.reshape(batch_size, shape[-2], shape[-1])
            
            ranks = torch.zeros(batch_size, dtype=torch.int64, device=M.device)
            for i in range(batch_size):
                ranks[i] = GF2Ops.rank(M_flat[i], validate=False)
            
            return ranks.reshape(batch_shape)
        
        # Single matrix
        m, n = M.shape
        A = M.to(torch.int8).clone()
        rank = 0
        
        for col in range(min(m, n)):
            # Find pivot in remaining rows
            pivot_found = False
            for row in range(rank, m):
                if A[row, col] == 1:
                    # Swap rows if needed
                    if row != rank:
                        A[[rank, row]] = A[[row, rank]]
                    pivot_found = True
                    break
            
            if not pivot_found:
                continue
            
            # Eliminate column
            for row in range(m):
                if row != rank and A[row, col] == 1:
                    A[row] ^= A[rank]
            
            rank += 1
        
        return rank
    
    @staticmethod
    def solve(A: torch.Tensor, b: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """Solve linear system Ax = b over GF(2)."""
        if validate:
            GF2Ops._validate_tensor(A)
            GF2Ops._validate_tensor(b)
        
        m, n = A.shape[-2:]
        device = A.device
        dtype = torch.int8
        
        # Prepare augmented matrix
        b_reshaped = b.reshape(m, -1) if b.dim() == 1 else b
        k = b_reshaped.shape[1]
        
        # Create copies
        A_copy = A.to(dtype).clone()
        b_copy = b_reshaped.to(dtype).clone()
        
        # Gauss-Jordan elimination on augmented matrix
        rank = 0
        pivot_cols = []
        
        for col in range(n):
            # Find pivot
            pivot_found = False
            for row in range(rank, m):
                if A_copy[row, col] == 1:
                    # Swap rows
                    if row != rank:
                        A_copy[[rank, row]] = A_copy[[row, rank]]
                        b_copy[[rank, row]] = b_copy[[row, rank]]
                    pivot_found = True
                    pivot_cols.append(col)
                    break
            
            if not pivot_found:
                continue
            
            # Eliminate column
            for row in range(m):
                if row != rank and A_copy[row, col] == 1:
                    A_copy[row] ^= A_copy[rank]
                    b_copy[row] ^= b_copy[rank]
            
            rank += 1
        
        # Check if system has solution
        for row in range(rank, m):
            if b_copy[row].any():
                raise RuntimeError("Linear system has no solution over GF(2)")
        
        # Back substitution
        x = torch.zeros((n, k), dtype=dtype, device=device)
        for i, col in enumerate(pivot_cols):
            x[col] = b_copy[i]
        
        # Return in original shape
        if b.dim() == 1:
            x = x.squeeze(-1)
        
        return x.bool() if A.dtype == torch.bool else x.to(A.dtype)
    
    @staticmethod
    def determinant(M: torch.Tensor, validate: bool = True) -> Union[int, torch.Tensor]:
        """Compute determinant over GF(2)."""
        if validate:
            GF2Ops._validate_tensor(M)
        
        if M.dim() > 2:
            # Batched determinant
            shape = M.shape
            n = shape[-1]
            if shape[-2] != n:
                raise ValueError("All matrices must be square")
            
            batch_shape = shape[:-2]
            batch_size = torch.prod(torch.tensor(batch_shape)).item()
            M_flat = M.reshape(batch_size, n, n)
            
            dets = torch.zeros(batch_size, dtype=torch.int32, device=M.device)
            for i in range(batch_size):
                dets[i] = GF2Ops.determinant(M_flat[i], validate=False)
            
            return dets.reshape(batch_shape)
        
        # Single matrix
        n = M.shape[0]
        if n != M.shape[1]:
            raise ValueError("Matrix must be square")
        
        if n == 1:
            return M[0, 0].item()
        
        if n == 2:
            return int((M[0, 0] * M[1, 1] + M[0, 1] * M[1, 0]) & 1)
        
        # For larger matrices, det = 1 iff full rank
        rank = GF2Ops.rank(M, validate=False)
        return 1 if rank == n else 0
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Convenience methods
    @staticmethod
    def batch_matmul(M: torch.Tensor, V: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """Batch matrix multiplication."""
        return GF2Ops.matmul(M, V, validate)
    
    @staticmethod
    def batch_invert(matrices: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """Batch matrix inversion."""
        return GF2Ops.invert_matrix(matrices, validate)
    
    @staticmethod
    def vstack(tensors: List[torch.Tensor], validate: bool = True) -> torch.Tensor:
        """Vertically stack tensors."""
        if validate:
            for t in tensors:
                GF2Ops._validate_tensor(t)
        
        dtype = tensors[0].dtype
        tensors_work = [t.to(torch.int8) for t in tensors]
        result = torch.vstack(tensors_work) & 1
        
        return result.bool() if dtype == torch.bool else result.to(dtype)
    
    @staticmethod
    def hstack(tensors: List[torch.Tensor], validate: bool = True) -> torch.Tensor:
        """Horizontally stack tensors."""
        if validate:
            for t in tensors:
                GF2Ops._validate_tensor(t)
        
        dtype = tensors[0].dtype
        tensors_work = [t.to(torch.int8) for t in tensors]
        result = torch.hstack(tensors_work) & 1
        
        return result.bool() if dtype == torch.bool else result.to(dtype)
    
    @staticmethod
    def concatenate(tensors: List[torch.Tensor], dim: int = 0, validate: bool = True) -> torch.Tensor:
        """Concatenate tensors along dimension."""
        if validate:
            for t in tensors:
                GF2Ops._validate_tensor(t)
        
        dtype = tensors[0].dtype
        tensors_work = [t.to(torch.int8) for t in tensors]
        result = torch.cat(tensors_work, dim=dim) & 1
        
        return result.bool() if dtype == torch.bool else result.to(dtype)
