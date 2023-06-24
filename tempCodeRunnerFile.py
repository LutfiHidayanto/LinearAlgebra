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