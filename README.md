# LinearAlgebra

# Matrix Calculator

A Python program that performs various matrix calculations, including solving linear equation systems, singular value decomposition, eigenvalues and eigenvectors, matrix inverses, checking if a matrix is diagonal, polynomial characteristics, and diagonalizing matrices.

## Requirements

- Python 3.x
- numpy library
- sympy library

## Installation

1. Clone the repository or download the program files.
2. Install the required libraries by running the following command:
   ```
   pip install numpy sympy
   ```

## Usage

1. Open a terminal or command prompt.
2. Navigate to the directory where the program files are located.
3. Run the program by executing the following command:
   ```
   python matrix_calculator.py
   ```
4. Follow the on-screen instructions to choose the desired operation and provide the necessary inputs.
5. View the results in the terminal or command prompt.
6. The program will also generate a file named "matrix_output.txt" that contains the calculations performed.

## Program Menu

The program provides the following options:

1. Linear Equation System: Solves a system of linear equations.
2. Singular Value Decomposition (SVD): Performs singular value decomposition on a matrix.
3. Eigenvalues and Eigenvectors: Calculates the eigenvalues and eigenvectors of a matrix.
4. Matrix Inverse: Computes the inverse of a matrix.
5. Check if Matrix is Diagonal: Checks if a matrix is diagonal.
6. Polynomial Characteristics: Calculates the polynomial characteristics of a matrix.
7. Diagonalize Matrix: Diagonalizes a matrix.
8. Complex Linear Equation System: Solves a system of complex linear equations.
9. Exit: Quits the program.

## Examples

1. Solving a system of linear equations:

   ```
   Choose solving method:
   1. Gauss-Jordan
   2. LU decomposition with partial pivoting
   3. Least squares
   Note: non-square matrices only work with the least squares method
   Choice: 1

   Input:
   Matrix A:
   [[1 2 3]
    [4 5 6]
    [7 8 9]]

   Matrix b: [1 2 3]

   Result:
   Solution: [-0.66667  1.33333  0.33333]
   ```

2. Performing singular value decomposition (SVD):

   ```
   Input:
   Matrix A:
   [[1 2]
    [3 4]
    [5 6]]

   Result:
   U:
   [[-0.22984 -0.88346]
    [-0.52472 -0.24078]
    [-0.81961  0.40189]]

   S:
   [9.52552  0.5143]

   V:
   [[-0.61963 -0.78489]
    [-0.78489  0.61963]]
   ```

3. Calculating eigenvalues and eigenvectors:

   ```
   Input:
   Matrix A:
   [[1 2]
    [3 4]]

   Result:
   Eigenvalues: [-0.37228  5.37228]

   Eigenvectors:
   [[-0.82456 -0.41597]
    [ 0.56577 -0.90938]]
   ```

4. Computing the matrix inverse:

   ```
   Input:
   Matrix A:
   [[1 2]
    [3 4]]

   Result:
   Inverse matrix:
   [[-2.   1. ]
    [ 1.5 -0.5]]
  

 ```

5. Checking if a matrix is diagonal:

   ```
   Input:
   Matrix A:
   [[1 0 0]
    [0 2 0]
    [0 0 3]]

   Result:
   Matrix A is diagonal.
   ```

6. Calculating polynomial characteristics:

   ```
   Input:
   Matrix A:
   [[1 2]
    [3 4]]

   Result:
   Characteristic polynomial: lambda^2 - 5*lambda - 2

   Eigenvalues: [5.37228 -0.37228]
   ```

7. Diagonalizing a matrix:

   ```
   Input:
   Matrix A:
   [[1 2]
    [3 4]]

   Result:
   Diagonal matrix D:
   [[-0.37228  0.     ]
    [ 0.       5.37228]]

   Diagonalizing matrix P:
   [[-0.90938 -0.41597]
    [ 0.56577 -0.82456]]

   Diagonal matrix D = P^(-1) * A * P
   ```

## License

This project is licensed under the [MIT License](LICENSE).
```

Please note that the format might look slightly different when viewed in a Markdown renderer, but the content should remain the same.
