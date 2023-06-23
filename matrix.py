import numpy as np
from sympy import symbols, Eq, solve

def write_to_file(title, output, *matrix):
    with open('matrix_output.txt', 'a') as file:
        file.write(f"\n{title}\n")
        char = 'A'
        for i in matrix:
            file.write(f"Matrix {char}:\n")
            char = chr(ord(char) + 1)
            file.write(str(i))
            file.write("\n\n")
        file.write("Output:\n")
        if isinstance(output, dict):
            for key, value in output.items():
                file.write(f"{key}:\n{value}\n")
        else:
            file.write(str(output) + "\n")


def gauss_jordan_elimination(A, b):
    n = A.shape[0]  # Number of equations
    m = A.shape[1]  # Number of variables

    # Convert matrix A and vector b into float data type from int
    A = A.astype(float)
    b = b.astype(float)

    # Augmented matrix [A|b]
    aug_matrix = np.concatenate((A, np.expand_dims(b, axis=1)), axis=1)

    # Apply Gauss-Jordan elimination
    for i in range(n):
        # Partial pivoting
        max_row = i
        for j in range(i + 1, n):
            if abs(aug_matrix[j, i]) > abs(aug_matrix[max_row, i]):
                max_row = j
        aug_matrix[[i, max_row], :] = aug_matrix[[max_row, i], :]

        if abs(aug_matrix[i, i]) < 1e-10:  # Check for zero pivot
            rank = np.linalg.matrix_rank(aug_matrix[:, :-1])
            # comparing rank with number of variables
            if rank < m:
                print("The system of equations is underdetermined with infinite solutions.")
                # Calculate the general solution
                general_solution = {}
                x = symbols('x1:%d' % (m + 1))
                equations = [Eq(aug_matrix[j, :-1].dot(x), aug_matrix[j, -1]) for j in range(rank)]
                solutions = solve(equations, x[:rank])
                for j in range(m):
                    if j < rank:
                        general_solution[f'x{j+1}'] = solutions[x[j]]
                    else:
                        general_solution[f'x{j+1}'] = x[j]  # Storing symbol like x1, x2, etc
                return general_solution
            else:
                print("The system of equations has no solution.")
            return None

        # Row operations
        aug_matrix[i, :] /= aug_matrix[i, i]
        for j in range(n):
            if j != i:
                aug_matrix[j, :] -= aug_matrix[j, i] * aug_matrix[i, :]

    # Forming solution
    x = aug_matrix[:, -1]

    solution = {}
    for i in range(m):
        solution[f'x{i+1}'] = x[i]

    return solution


def solve_linear_equation(A, b):
    # Ax = b
    # A: coefficient matrix
    # b: constant vector
    print("Choose solving method")
    print("1. Gauss-Jordan")
    print("2. LU decomposition with partial pivoting")
    print("3. Least squares")
    print("Note: non-square matrices only work with least squares method")
    choice = int(input("Choice: "))
    if choice == 1:
        x = gauss_jordan_elimination(A, b)
        if x is not None:
            print("Using Gauss-Jordan Method")
            print(f"Solution:\n {x}")
    elif choice == 2:
        try:
            print("Using LU decomposition with partial pivoting")
            x = np.linalg.solve(A, b)
            print(f"Solution: {x}")
        # Catch exception if matrix is non square, or there is no solution or infinite
        except np.linalg.LinAlgError:
            print("The system of equations has no solution or underdetermined")
            return None
    # For non square matrix
    elif choice == 3:
        print("Using Least Square Approach:")
        # solving problem using least square and store the results in x
        x, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
        # comparing rank with number of variables
        if rank < A.shape[1]: 
            print("The system of equations has no solution.")
            return None
        elif rank < A.shape[1]:
            print("The system of equations is underdetermined with infinite solutions.")
            return None
        else:
            print("The system of equations has a unique solution.")
            print(f"Solution: {x}")
    # if input is invalid
    else:
        return None
    return x

def singular_value_decomposition(matrix):
    # calculate SVD 
    U, S, V = np.linalg.svd(matrix)
    print(f"U:\n {U}\nS:\n {S}\nV:\n {V}")
    return {'U': U, 'S': S, 'V': V}

def solve_complex_linear_equation(A, b):
    # Singular Value Decomposition
    # if matrix is square then use SVD method
    if A.shape[0] == A.shape[1]:
        print("Using SVD method")
        U, s, Vh = np.linalg.svd(A)
        # Pseudoinverse of Σ
        S_inv = np.zeros(A.shape, dtype=complex)
        S_inv[:A.shape[1], :A.shape[1]] = np.diag(1/s)

        # Compute the solution matrix X
        X = Vh.conj().T @ S_inv @ U.conj().T @ b[:, np.newaxis]
        X = X.flatten()
    # if matrix non square, use moore-penrose 
    else:
        print("Using Moore-Penrose pseudoinverse")
        A_inv = np.linalg.pinv(A)
        X = np.dot(A_inv, b)
    print(X)
    return X

def is_diagonal(matrix):
    # Check if a matrix is diagonalized 
    eigenvalues, _ = np.linalg.eig(matrix)
    is_diagonal = np.allclose(matrix, np.diag(eigenvalues))
    if is_diagonal:
        print(f"Matrix is diagonal")
    else:
        print("Matrix is not diagonal")
    return is_diagonal

def eigenvalues_vector(matrix):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvector  = np.linalg.eig(matrix)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvector:\n {eigenvector}")
    return {"Eigenvalues": eigenvalues, "Eigenvector": eigenvector}

def polynomial_char(matrix):
    # calculate polynomaial characteristics
    eigenvalues = np.linalg.eigvals(matrix)
    polychar = np.poly(eigenvalues)
    print(f"Polynomial Characteristics:\n {polychar}")
    return polychar

def invers(matrix):
    # calculate matrix invers
    invers = np.linalg.inv(matrix)
    print(f"Invers matrix:\n {invers}")
    return invers

def diagonalize_matrix(A):
    # calculate the eigendecomposition of A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # construct the diagonal matrix D with eigenvalues
    D = np.diag(eigenvalues)
    # Construct the matrix P with eigenvectors as columns
    P = eigenvectors
    # calculate the inverse of P
    P_inv = np.linalg.inv(P)
    # calculate P^(-1)AP
    diagonalized_A = P_inv @ A @ P
    print(f"Matrix P:\n {P}") 
    print(f"Diagonal D:\n {D}")
    print(F"Diagonalized A:\n {diagonalized_A}")
    return {"P": P, "D": D, "Diagonalized A": diagonalized_A}


def twod_input():
    # Handling 2 dimension input matrix 
    print("Enter matrix A")
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    # Initialize an empty list to store the input values
    input_values = []

    # Get input values from the user and append them to the list
    for _ in range(rows):
        row_input = input("Enter space-separated values for a row: ")
        values = [int(value) for value in row_input.split()]
        input_values.append(values)
    array_2d = np.array(input_values)

    return array_2d


def complex_twod_input():
    print("Enter Complex matrix A")
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    # Initialize an empty list to store the input values
    input_values = []

    # Get input values from the user and append them to the list
    for _ in range(rows):
        row_input = input("Enter space-separated values for a row: ")
        values = [complex(value.replace('-j', '-1j')) for value in row_input.split()]
        input_values.append(values)
    array_2d = np.array(input_values)

    return array_2d


def main():
    new_matrix = 'Y'
    while(True):
        print("PROGRAM MENU")
        print("1. LINEAR EQUATIONS SYSTEM")
        print("2. SVD")
        print("3. EIGENVALUE & EIGENVECTOR")
        print("4. INVERS")
        print("5. CHECK IF DIAGONAL")
        print("6. POLYNOMIAL CHARACTERISTICS")
        print("7. DIAGONALIZE MATRIX")
        print("8. COMPLEX LINEAR EQUATIONS SYSTEM")
        print("9. EXIT")
        choice = int(input("INPUT: "))
        b = None
        if (new_matrix == 'Y' or new_matrix == 'y') and choice != 8:
            a = twod_input()
        elif choice == 8:
            a = complex_twod_input()
        if choice != 8:
            print("\nInput:")
            print(f"Matrix A:\n {a}\n")
        if choice != 1 and choice != 8:
            print("\nResult: ")

        if choice == 1:
            b = np.array([int(value) for value in input("Enter matrix b: ").split()])
            print("\nResult: ")
            sol = solve_linear_equation(a, b)
            title = "Linear Equation System"
        elif choice == 2:
            sol = singular_value_decomposition(a)
            title = "Singular Value Decomposition"
        elif choice == 3:
            sol = eigenvalues_vector(a)
            title = "Eigenvalues and Eigenvectors"
        elif choice == 4:
            sol = invers(a)
            title = "Invers"
        elif choice == 5:
            sol = is_diagonal(a)
            title = "Is Matrix Diagonal?"
        elif choice == 6:
            sol = polynomial_char(a)
            title = "Polynomial Characteristics"
        elif choice == 7:
            sol = diagonalize_matrix(a)
            title = "Diagonalize Matrix"
        elif choice == 8:
            b = np.array([complex(value) for value in input("Enter matrix b: ").split()])
            print("\nResult: ")
            sol = solve_complex_linear_equation(a, b)
            title = "Complex Linear Equations System"
        elif choice == 9:
            break
        else:
            print("Input is not valid. Try again!")
        if b is not None:
            write_to_file(title, sol, a, b)
        else:
            write_to_file(title, sol, a)
        while True:
            new_matrix = (input("Do you want to input a new matrix?(Y/N)"))
            if new_matrix != 'Y' and new_matrix != 'y' and new_matrix != 'N' and new_matrix != 'n':
                print("Invalid input. Try again!")
            else:
                break
        print("\n")

if __name__ == "__main__":
    main()


