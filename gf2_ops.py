import torch
import abc
from typing import List, Optional

class AbstractGF2Ops(abc.ABC):
    """
    Abstract Base Class for GF(2) operations.

    Defines the interface for GF(2) matrix multiplication and inversion.
    """

    @staticmethod
    @abc.abstractmethod
    def matmul(M: torch.Tensor, v: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Compute the matrix-vector product over GF(2) as (M @ v) mod 2.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def invert_matrix(M_in: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Compute the inverse of a square matrix over GF(2) using Gauss–Jordan elimination.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add(A: torch.Tensor, B: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Compute (A + B) mod 2 elementwise.
        """
        pass


class GF2Ops(AbstractGF2Ops):
    """
    Implementation of GF(2) operations for efficient arithmetic in the binary field.

    This class provides efficient implementations for matrix operations over GF(2),
    including matrix multiplication, inversion, and addition. It supports batch operations
    and various input tensor types.
    """

    # Class-level attribute for floating-point comparison
    _EPSILON: float = 1e-6

    @staticmethod
    def _validate_tensor(tensor: torch.Tensor) -> None:
        """
        Validates that tensor elements are in {0, 1}. Accepts float32/float64, integer types, or bool.
        Raises a ValueError if any element is not 0 or 1.
        """
        # Fast path for boolean tensors
        if tensor.dtype == torch.bool:
            return

        # Fast path for common integer types
        if tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            invalid = ~((tensor == 0) | (tensor == 1))
            if invalid.any():
                raise ValueError("Tensor of integer type must only contain 0 or 1.")
            return

        # Check floating point tensors
        if tensor.dtype in [torch.float32, torch.float64]:
            # Use a small epsilon for floating point comparison
            epsilon = GF2Ops._EPSILON
            invalid = ~(((tensor - 0.0).abs() < epsilon) | ((tensor - 1.0).abs() < epsilon))
            if invalid.any():
                raise ValueError(
                    "Tensor of float type must only contain values approximately 0.0 or 1.0."
                )
            return

        raise TypeError(f"Unsupported tensor type {tensor.dtype} for GF(2) operations.")

    @staticmethod
    def _batch_validate(tensors: List[torch.Tensor]) -> None:
        """
        Validates multiple tensors at once.

        Args:
            tensors: List of torch.Tensor objects to validate
        """
        for tensor in tensors:
            GF2Ops._validate_tensor(tensor)

    @staticmethod
    def matmul(M: torch.Tensor, v: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Computes GF(2) matrix multiplication: result = (M @ v) mod 2.
        Optimized for different input types and shapes.

        Args:
            M: Matrix tensor of shape (m, n)
            v: Vector/matrix tensor of shape (n, k) or (n,)
            validate: Whether to validate inputs

        Returns:
            Result tensor of shape (m, k) or (m,)
        """
        if validate:
            GF2Ops._batch_validate([M, v])

        # Check if any dimension is very large, in which case we'll use an optimized approach
        if M.shape[0] > 1000 or (v.dim() > 1 and v.shape[1] > 1000):
            return GF2Ops._large_matmul(M, v)

        # Boolean tensors optimization
        if M.dtype == torch.bool and v.dtype == torch.bool:
            # For boolean inputs, we can use a specialized approach avoiding float conversion
            M_int = M.to(torch.int32)
            v_int = v.to(torch.int32)
            
            # Use int32 multiplication and modulo for better performance
            result = torch.matmul(M_int, v_int) % 2
            return result.bool()

        # Integer type optimization
        is_M_int = M.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        is_v_int = v.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        
        # Convert only if needed
        M_int = M if is_M_int else M.to(torch.int32)
        v_int = v if is_v_int else v.to(torch.int32)
            
        # Use binary operations directly if possible for better performance
        result = torch.matmul(M_int, v_int) % 2
            
        # Return in appropriate dtype
        if M.dtype == torch.bool or v.dtype == torch.bool:
            return result.bool()
        return result

    @staticmethod
    def _large_matmul(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Optimized matrix multiplication for large matrices in GF(2).
        Uses block processing to reduce memory usage for very large matrices.
        
        Args:
            M: Matrix tensor of shape (m, n)
            v: Vector/matrix tensor of shape (n, k) or (n,)
            
        Returns:
            Result tensor of shape (m, k) or (m,)
        """
        # Convert to most efficient integer type for operations
        M_int = M.to(torch.int32) if M.dtype != torch.int32 else M
        v_int = v.to(torch.int32) if v.dtype != torch.int32 else v
        
        # Ensure v is a matrix
        if v_int.dim() == 1:
            v_int = v_int.unsqueeze(1)
            
        m, n = M_int.shape
        _, k = v_int.shape
        
        # Choose block size based on matrix dimensions
        # This helps with memory usage for very large matrices
        block_size = min(1024, m)
        
        result = torch.zeros((m, k), dtype=torch.int32, device=M.device)
        
        # Process matrix multiplication in blocks
        for i in range(0, m, block_size):
            end_idx = min(i + block_size, m)
            # Process a block of the matrix
            result[i:end_idx] = torch.matmul(M_int[i:end_idx], v_int) % 2
            
        # Return result in the appropriate type
        if M.dtype == torch.bool or v.dtype == torch.bool:
            return result.bool()
        return result

    @staticmethod
    def batch_matmul(M: torch.Tensor, V: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Computes batched GF(2) matrix multiplication.

        Args:
            M: Tensor of shape (..., m, n)
            V: Tensor of shape (..., n, k)
            validate: Whether to validate inputs

        Returns:
            Result tensor of shape (..., m, k)
        """
        if validate:
            GF2Ops._batch_validate([M, V])

        if M.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            M_int = M
        else:
            M_int = M.to(torch.int32)

        if V.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            V_int = V
        else:
            V_int = V.to(torch.int32)

        # Perform batched matrix multiplication
        result = torch.bmm(
            M_int.view(-1, *M_int.shape[-2:]),
            V_int.view(-1, *V_int.shape[-2:])
        )
        result = (result % 2).view(*M.shape[:-2], M.shape[-2], V.shape[-1])

        if M.dtype == torch.bool and V.dtype == torch.bool:
            return result.bool()
        return result

    @staticmethod
    def invert_matrix(M_in: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Inverts a square matrix M_in over GF(2) using optimized Gauss–Jordan elimination.

        Args:
            M_in: Square matrix tensor of shape (n, n)
            validate: Whether to validate the input

        Returns:
            Inverted matrix as a boolean tensor

        Raises:
            ValueError: If M_in is not square
            RuntimeError: If M_in is not invertible over GF(2)
        """
        if validate:
            GF2Ops._validate_tensor(M_in)

        if M_in.shape[0] != M_in.shape[1]:
            raise ValueError("Matrix must be square to invert.")

        n = M_in.shape[0]
        device = M_in.device

        # Fast path for small matrices
        if n <= 4:
            return GF2Ops._small_matrix_inverse(M_in.to(torch.int32).clone(), 
                                               torch.eye(n, dtype=torch.int32, device=device), n)

        # Work on integer copies for better performance
        A = M_in.to(torch.int32).clone()
        I = torch.eye(n, dtype=torch.int32, device=device)
        
        # Use Gauss-Jordan elimination
        rank = GF2Ops._gauss_jordan_elimination(A, I)
        
        # Check if matrix is invertible
        if rank < n:
            raise RuntimeError("Matrix is not invertible over GF(2).")
            
        return I.bool()

    @staticmethod
    def _small_matrix_inverse(A: torch.Tensor, I: torch.Tensor, n: int) -> torch.Tensor:
        """
        Optimized inversion for small matrices (n <= 4).
        """
        for i in range(n):
            # Direct pivoting
            if A[i, i] == 0:
                found = False
                for j in range(i + 1, n):
                    if A[j, i] == 1:
                        A[[i, j]] = A[[j, i]]
                        I[[i, j]] = I[[j, i]]
                        found = True
                        break
                if not found:
                    raise RuntimeError("Small matrix is not invertible over GF(2).")

            # Elimination
            for j in range(n):
                if j != i and A[j, i] == 1:
                    A[j] = torch.bitwise_xor(A[j], A[i])
                    I[j] = torch.bitwise_xor(I[j], I[i])

        return I.bool()

    @staticmethod
    def batch_invert(matrices: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Inverts multiple square matrices over GF(2) in batch.

        Args:
            matrices: Tensor of shape (b, n, n) containing b matrices to invert
            validate: Whether to validate inputs

        Returns:
            Tensor of shape (b, n, n) containing the inverted matrices
        """
        if validate:
            GF2Ops._validate_tensor(matrices)

        batch_size, n, m = matrices.shape
        if n != m:
            raise ValueError("All matrices must be square to invert.")

        device = matrices.device

        # Work on integer copies
        As = matrices.to(torch.int32).clone()
        Is = torch.eye(n, dtype=torch.int32, device=device).expand(batch_size, n, n).clone()

        # Process each matrix in the batch
        for b in range(batch_size):
            A = As[b]
            I = Is[b]

            for i in range(n):
                # Find pivot
                if A[i, i] == 0:
                    non_zero_rows = torch.nonzero(A[i:, i], as_tuple=False).view(-1)
                    if len(non_zero_rows) == 0:
                        raise RuntimeError(
                            f"Matrix at index {b} is not invertible over GF(2)."
                        )

                    swap_row = i + non_zero_rows[0].item()
                    A[[i, swap_row]] = A[[swap_row, i]]
                    I[[i, swap_row]] = I[[swap_row, i]]

                rows_to_update = torch.nonzero(A[:, i], as_tuple=False).view(-1)
                rows_to_update = rows_to_update[rows_to_update != i]

                if len(rows_to_update) > 0:
                    A[rows_to_update] = torch.bitwise_xor(A[rows_to_update], A[i])
                    I[rows_to_update] = torch.bitwise_xor(I[rows_to_update], I[i])

            Is[b] = I

        return Is.bool()

    @staticmethod
    def add(A: torch.Tensor, B: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Elementwise GF(2) addition: (A + B) mod 2.
        Optimized for various tensor types.

        Args:
            A: First tensor
            B: Second tensor with same shape as A
            validate: Whether to validate inputs

        Returns:
            Result of A + B over GF(2)
        """
        if validate:
            GF2Ops._batch_validate([A, B])

        if A.shape != B.shape:
            raise ValueError(
                f"Tensors must have the same shape for addition. Got {A.shape} and {B.shape}."
            )

        # Fast path for boolean tensors
        if A.dtype == torch.bool and B.dtype == torch.bool:
            return torch.logical_xor(A, B)

        # Otherwise, convert to integer if necessary and use bitwise_xor
        if A.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            A_int = A.to(torch.int32)
        else:
            A_int = A

        if B.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            B_int = B.to(torch.int32)
        else:
            B_int = B

        result_int = torch.bitwise_xor(A_int, B_int)
        # Return result cast back to original dtype of A
        return result_int.to(A.dtype)

    @staticmethod
    def rank(M: torch.Tensor, validate: bool = True) -> int:
        """
        Computes the rank of a matrix over GF(2).

        Args:
            M: Matrix tensor
            validate: Whether to validate the input

        Returns:
            Rank of the matrix over GF(2)
        """
        if validate:
            GF2Ops._validate_tensor(M)

        # Convert to int32 for elimination
        A = M.to(torch.int32).clone()
        
        # Use Gauss-Jordan elimination to compute rank
        return GF2Ops._gauss_jordan_elimination(A)

    @staticmethod
    def solve(A: torch.Tensor, b: torch.Tensor, validate: bool = True) -> torch.Tensor:
        """
        Solves the linear system Ax = b over GF(2).

        Args:
            A: Coefficient matrix of shape (m, n)
            b: Right-hand side vector/matrix of shape (m,) or (m, k)
            validate: Whether to validate inputs

        Returns:
            Solution vector/matrix of shape (n,) or (n, k) if solution exists

        Raises:
            RuntimeError: If no solution exists
        """
        if validate:
            GF2Ops._batch_validate([A, b])

        m, n = A.shape
        if b.shape[0] != m:
            raise ValueError(
                f"Shape mismatch: A has {m} rows but b has {b.shape[0]} rows."
            )

        device = A.device

        # Convert b to shape (m, k)
        b_reshaped = b.reshape(m, -1) if b.dim() == 1 else b
        k = b_reshaped.shape[1]

        # Create copies for Gauss-Jordan elimination
        A_copy = A.to(torch.int32).clone()
        b_copy = b_reshaped.to(torch.int32).clone()
        
        # Use Gauss-Jordan elimination
        rank_A = GF2Ops._gauss_jordan_elimination(A_copy, b_copy)
        
        # Check if system has a solution
        for i in range(rank_A, m):
            if torch.any(b_copy[i] != 0):
                raise RuntimeError("Linear system has no solution over GF(2).")
                
        # Back-substitution to find solution
        x = torch.zeros((n, k), dtype=torch.int32, device=device)
        
        # Process rows with pivots
        for i in range(min(rank_A, n)):
            # Find pivot column
            pivot_col = -1
            for j in range(n):
                if A_copy[i, j] == 1:
                    pivot_col = j
                    break
            
            if pivot_col >= 0:
                x[pivot_col] = b_copy[i]
        
        # Return boolean if original was boolean
        return x.bool() if A.dtype == torch.bool else x

    @staticmethod
    def determinant(M: torch.Tensor, validate: bool = True) -> int:
        """
        Computes determinant of a square matrix over GF(2).

        Args:
            M: Square matrix tensor
            validate: Whether to validate the input

        Returns:
            0 or 1 (the determinant over GF(2))
        """
        if validate:
            GF2Ops._validate_tensor(M)

        n = M.shape[0]
        if n != M.shape[1]:
            raise ValueError("Matrix must be square to compute determinant.")

        if n == 1:
            return M[0, 0].item()

        if n == 2:
            return (M[0, 0] * M[1, 1] + M[0, 1] * M[1, 0]) % 2

        # For larger matrices, use Gauss-Jordan elimination
        A = M.to(torch.int32).clone()
        rank = GF2Ops._gauss_jordan_elimination(A)
        
        # In GF(2), the determinant is 1 if the matrix is full rank, 0 otherwise
        return 1 if rank == n else 0
    
    @staticmethod
    def _gauss_jordan_elimination(A: torch.Tensor, B: Optional[torch.Tensor] = None) -> int:
        """
        Performs Gauss-Jordan elimination in-place on integer tensor A (mod 2).
        If B is provided, treats [A|B] as an augmented matrix.
        Returns the rank of A after elimination.

        A and B should be torch.int32 or a similar integer type.
        """
        m, n = A.shape
        rank_ = 0
        
        for col in range(n):
            # Find the pivot row
            non_zero_rows = torch.nonzero(A[rank_:, col], as_tuple=False).view(-1)
            if len(non_zero_rows) == 0:
                continue  # No pivot in this column
                
            # Get the pivot row index
            pivot_row = rank_ + non_zero_rows[0].item()
            
            # Swap pivot row with current row if needed
            if pivot_row != rank_:
                A[[rank_, pivot_row]] = A[[pivot_row, rank_]]
                if B is not None:
                    B[[rank_, pivot_row]] = B[[pivot_row, rank_]]
            
            # Find rows that need elimination (rows with 1 in current column, except pivot row)
            rows_to_eliminate = torch.nonzero(A[:, col], as_tuple=False).view(-1)
            rows_to_eliminate = rows_to_eliminate[rows_to_eliminate != rank_]
            
            # Vectorized elimination for all rows that need it
            if len(rows_to_eliminate) > 0:
                A[rows_to_eliminate] = torch.bitwise_xor(A[rows_to_eliminate], A[rank_].unsqueeze(0).expand(len(rows_to_eliminate), -1))
                if B is not None:
                    B[rows_to_eliminate] = torch.bitwise_xor(B[rows_to_eliminate], B[rank_].unsqueeze(0).expand(len(rows_to_eliminate), -1))
            
            rank_ += 1
            if rank_ == m:
                break
                
        return rank_