import numpy as np

def stack_and_concatenate(matrix, L):
    rows, cols = matrix.shape
    print(rows, cols)
    result = []

    for k in range(rows//L):
        y = matrix[k * L:(k + 1) * L, :]
        stacked_col = y.T.reshape(1, -1).T
        result.append(stacked_col)

    return np.hstack(result)

def concatenate_and_stack(matrix, L):
    rows, cols = matrix.shape
    result = []

    for j in range(cols):
        y = matrix[:, j]
        stacked_row = y.T.reshape(-1, L).T
        result.append(stacked_row)

    return np.vstack(result)

# Example usage:
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

L = 2
result = stack_and_concatenate(matrix, L)
print("Result:")
print(result)

result = concatenate_and_stack(result, L)
print("Result:")
print(result)