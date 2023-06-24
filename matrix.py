import numpy as np
from sympy import symbols, Eq, solve

DECIMAL_PLACES = 5

def write_to_file(title, output, *matrix):
    # For writing calculations into txt file
    with open('matrix_output.txt', 'a') as file:
        # writing title of the calculation
        file.write(f"\n{title}\n")
        char = 'A'
        # writing matrix input
        for i in matrix:
            file.write(f"Matrix {char}:\n")
            # increase ascii value of A, so it become B
            char = chr(ord(char) + 1) 
            file.write(str(i))
            file.write("\n\n")
        file.write("Output:\n")
        # writing solution
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

    if n != m:
        print("Can't calculate matrix is non square with this method")
        return None

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


def solve_linear_equation(A, b, rounded = False):
    # Ax = b
    # A: coefficient matrix
    # b: constant vector
    print("Choose solving method")
    print("1. Gauss-Jordan")
    print("2. LU decomposition with partial pivoting")
    print("3. Least squares")
    print("Note: non-square matrices only work with least squares method")
    choice = int(input("Choice: "))
    print("\nResult: ")
    if choice == 1:
        x = gauss_jordan_elimination(A, b)
        if x is not None:
            print("Using Gauss-Jordan Method")
    elif choice == 2:
        try:
            print("Using LU decomposition with partial pivoting")
            x = np.linalg.solve(A, b)
        # Catch exception if matrix is non square, or there is no solution or infinite
        except np.linalg.LinAlgError:
            print("The system of equations has no solution or underdetermined")
            return None
    # For non square matrix
    elif choice == 3:
        print("Using Least Square Approach:")
        # solving problem using least square and store the results in x
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    # if input is invalid
    else:
        return None
    # printing solution for choice 2, 3 
    if x is not None and choice != 1:
        x = [round(num, DECIMAL_PLACES) for num in x]
    print(f"Solution: {x}")
    return x

def singular_value_decomposition(matrix, rounded = False):
    # calculate SVD 
    try:
        U, S, V = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        print("Can't compute this matrix")
        return None
    if rounded:
        U = np.round(U, DECIMAL_PLACES)
        S = np.round(S, DECIMAL_PLACES)
        V = np.round(V, DECIMAL_PLACES)
    print(f"U:\n {U}\nS:\n {S}\nV:\n {V}")
    return {'U': U, 'S': S, 'V': V}

def solve_complex_linear_equation(A, b, rounded=False):
    # Singular Value Decomposition
    U, S, V = np.linalg.svd(A, full_matrices=False)
    X = V.T @ np.diag(1 / S) @ U.T @ b
    if rounded:
        X = np.round(X, DECIMAL_PLACES)
    solution = {}
    # Change the solution into dictionary so that it more readable
    for i, value in enumerate(X):
        solution[f'x{i+1}'] = value
    print(f"Solution:\n{solution}")
    return solution


def is_diagonal(matrix):
    # Check if a matrix is diagonalized 
    eigenvalues, _ = np.linalg.eig(matrix)
    is_diagonal = np.allclose(matrix, np.diag(eigenvalues))
    if is_diagonal:
        print(f"Matrix is diagonal")
    else:
        print("Matrix is not diagonal")
    return is_diagonal

def eigenvalues_vector(matrix, rounded = False):
    # Calculate eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvector  = np.linalg.eig(matrix)
    except np.linalg.LinAlgError:
        print("Matrix must be Square")
        return None
    if rounded:
        eigenvalues = np.round(eigenvalues, DECIMAL_PLACES)
        eigenvector = np.round(eigenvector, DECIMAL_PLACES)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvector:\n{eigenvector}")
    return {"Eigenvalues": eigenvalues, "Eigenvector": eigenvector}

def polynomial_char(matrix, rounded = False):
    # calculate polynomaial characteristics
    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        print("Matrix must be Square")
        return None
    polychar = np.poly(eigenvalues)
    if rounded:
        polychar = np.round(polychar, DECIMAL_PLACES)
    print(f"Polynomial Characteristics:\n{polychar}")
    return polychar

def invers(matrix, rounded = False):
    # calculate matrix invers
    try:
        invers = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        print("Matrix must be Square")
        return None 
    if rounded:
        invers = np.round(invers, DECIMAL_PLACES)
    print(f"Invers matrix:\n{invers}")
    return invers

def diagonalize_matrix(A, rounded = False):
    # calculate the eigendecomposition of A
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        print("Matrix must be Square")
        return None 
    # construct the diagonal matrix D with eigenvalues
    D = np.diag(eigenvalues)
    # Construct the matrix P with eigenvectors as columns
    P = eigenvectors
    # calculate the inverse of P
    P_inv = np.linalg.inv(P)
    # calculate P^(-1)AP
    diagonalized_A = P_inv @ A @ P
    # Specify the desired number of decimal places
    if rounded:
        P = np.round(P, DECIMAL_PLACES)
        D = np.round(D, DECIMAL_PLACES)
        diagonalized_A = np.round(diagonalized_A, DECIMAL_PLACES)
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

def validate_user_input(inputstr):
    while True:
        user_input = input(inputstr)
        if user_input.lower() in ['y', 'n']:
            return user_input.lower()
        else:
            print("Invalid input. Try again")

def main():
    new_matrix = 'Y'
    while(True):
        # Program Menu
        print("======== Matrix Calculator ======\n")
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
        # Round number or not
        if choice != 5 and choice != 9:
            rounded = validate_user_input("Do you want to round number to 5 decimal places?:{Y/N} ")
            if rounded == 'y':
                rounded = True
            else:
                rounded = False 
        # If user wants to insert new matrix
        if (new_matrix == 'Y' or new_matrix == 'y') and choice != 8 and choice != 9: # and not a complex matrix
            a = twod_input()
        elif choice == 8 and choice != 9: # if user wants to insert new complex matrix 
            a = complex_twod_input()
        if choice != 8 and choice != 9:
            print("\nInput:")
            print(f"Matrix A:\n{a}\n")
        if choice != 1 and choice != 8 and choice != 9:
            print("\nResult:")
        # Processing matrix
        if choice == 1:
            if (new_matrix == 'Y'or new_matrix == 'y'):
                b = np.array([int(value) for value in input("Enter matrix b: ").split()])
            print(f"Matrix b: {b}\n")
            sol = solve_linear_equation(a, b, rounded)
            title = "Linear Equation System"
        elif choice == 2:
            sol = singular_value_decomposition(a, rounded)
            title = "Singular Value Decomposition"
        elif choice == 3:
            sol = eigenvalues_vector(a, rounded)
            title = "Eigenvalues and Eigenvectors"
        elif choice == 4:
            sol = invers(a, rounded)
            title = "Invers"
        elif choice == 5:
            sol = is_diagonal(a)
            title = "Is Matrix Diagonal?"
        elif choice == 6:
            sol = polynomial_char(a, rounded)
            title = "Polynomial Characteristics"
        elif choice == 7:
            sol = diagonalize_matrix(a, rounded)
            title = "Diagonalize Matrix"
        elif choice == 8:
            if (new_matrix == 'Y'or new_matrix == 'y'):
                b = np.array([complex(value) for value in input("Enter matrix b: ").split()])
            print(f"Matrix b: {b}\n")
            print("\nResult: ")
            sol = solve_complex_linear_equation(a, b, rounded)
            title = "Complex Linear Equations System"
        elif choice == 9:
            break
        else:
            print("Input is not valid. Try again!")
        # Writing to file txt
        if choice == 1 or choice == 8:
            write_to_file(title, sol, a, b)
        else:
            write_to_file(title, sol, a)
        # New Matrix or no
        new_matrix = validate_user_input("Do you want to input a new matrix?(Y/N)")
        print("\n")

if __name__ == "__main__":
    main()


