[![Build and Deploy Javadoc](https://github.com/jacobdwatters/Flag4j/actions/workflows/javadoc-gh-pages.yml/badge.svg)](https://github.com/jacobdwatters/Flag4j/actions/workflows/javadoc-gh-pages.yml)
<!-- TODO: Add coverage badge -->
<center>
<img src="figures/flag4j_primary_logo_variant.svg" style="max-height: 350px; width: auto;">
</center>

# Flag4j – Fast Linear Algebra for Java

Flag4j is a fast and easy-to-use linear algebra library for Java
and provides operations and linear algebra routines for real/complex sparse/dense tensors,
matrices, and vectors.

## State of the Project as of 7-DEC-2024

Flag4j is currently in the final steps of development before an initial beta release.
Nearly all features have been fully implemented or beta implementations.

### Recent Updates:

- I/O Improvements:
    - All tensors, vectors, and matrices may be serialized.
    - Added support for reading/writing from/to Matrix Market Exchange file formats.
        - Currently only supports a subset including real/complex/integer dense general matrices
          and real/complex/integer coordinate general matrices
    - Added support for reading/writing from/to csv files for real and complex matrices (i.e. `Matrix` and `CMatrix`).
- Improved API: There have been significant changes to the API, specifically with the inheritance hierarchy of arrays (tensors,
  matrices, vectors). This was in part to serve the creation of general field/ring/semiring tensors.
- New Algebraic Structures and Generic Tensors:
    - Added `Field`, `Ring`, and `Semiring` interfaces representing the mathematical objects.
        - Complex numbers are now implemented as a `Field`. There are now 64-bit and 128-bit variants of complex numbers.
        - Several reference implementations of the interfaces are provided, e.g. `Bool`, `RealInt16`, `Complex128`.
    - Added generic `FieldMatrix<T extends Field<T>>` for creating matrices for a generic field. This allows users
      to easily create a matrix for a custom field which implements the `Field` interface. Support for generic
      ring and semiring matrices are planned.

___

## Features and Functionality

### Algebraic Objects

- Complex numbers, fields, rings, and semirings.
- Vectors
    - Real/complex/field dense vector
    - Real/complex/field sparse vector
- Matrices
    - Dense Matrices
        - Real/complex/field dense dense matrix
    - Sparse Matrices
        - Real/complex/field COO matrix
        - Real/complex/field CSR matrix
        - Real/complex/field symmetric tri-diagonal matrix.
        - Permutation matrix
- Tensors
    - Real/complex/field dense tensor
    - Real/complex/field sparse COO tensor

### Operations

- Basic Arithmetic Operations: add, subtract, scalar/element multiply, scalar/element divide, etc.
- Basic Properties: tensor/matrix/vector shape, non-zero entries, etc.
- Basic Manipulations: insert, join/stack/augment, extract, etc.
- Basic Comparisons: equal, same shape, etc.
- Vector Operations:
    - Arithmetic: inner/outer/cross products, vector norms, etc.
    - Comparisons: Parallel, orthogonal, etc.
- Matrix Operations:
    - Arithmetic: Matrix multiplication, transpose, matrix norms, inverse, pseudo-inverse, etc.
    - Features: Matrix rank, symmetric, definiteness, eigenvalues and vectors, singularity, triangular, etc.
    - Comparisons: Similar, etc.
- Tensor Operations:
    - Arithmetic: Tensor dot product, tensor transpose, tensor inverse, etc.
    - Comparisons: Tensor rank/dimension, etc.

### Matrix Decompositions

- LU Decompositions (real/complex)
    - No pivoting
    - Partial Pivoting
    - Full Pivoting
- QR Decomposition (real/complex)
    - Householder Reflectors
    - Full/Reduced
- Cholesky Decomposition (real/complex)
- Hessenburg Decomposition (real/complex)
    - General and specialized symmetric and Hermitian implementations are provided.
- Eigen/Schur Decompositions (real/complex)
- Singular Value Decomposition (real/complex)

### Linear Solvers

- Exact solution for well-determined matrix systems
    - General systems
    - Triangular systems
- Exact solution for well-determined tensor or matrix equations
- Least Squares solution

### Linear and Homography Transformations

- Scale
- Shift
- Rotate
- Affine
- Projections
    - Orthographic
    - Perspective

### Random

- Random Complex Numbers
- Random Matrices
    - Orthogonal, Unitary, symmetric, triangular, etc.
- Random Vectors
- Random Tensors

### I/O: Read/write tensors, matrices, and vectors to/from a file.

- Serialization (all types)
- Matrix Market Exchange Format (all real/complex dense and sparse matrix types)
- CSV format (only supported for `Matrix` and `CMatrix`)

> [!NOTE]
> **AI-Assisted Development**:<br>
> Parts of Flag4j were partially generated using AI coding assistants. All AI-generated code has been carefully and manually reviewed,
> tested, and modified by human maintainers to ensure functionality and safety.
