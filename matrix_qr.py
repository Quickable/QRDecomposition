import numpy as np
import random

# tolerance for convergence when performing
# qr iterations
TOLERANCE = 10 ** -5
# maximum value in randomly generated array
MAX_VALUE = 100


def qr_decomposition(u):
    dimension = (u.shape[:2])[0]
    q = np.zeros((dimension, dimension))
    r = np.zeros((dimension, dimension))
    for i in range(dimension):
        q[:, i] = u[:, i]
        for j in range(i):
            r[j, i] = np.dot(q[:, j].T, u[:, i])
            q[:, i] = q[:, i] - np.dot(r[j, i], q[:, j])
        r[i, i] = np.linalg.norm(q[:, i], ord=2)
        q[:, i] = q[:, i] / r[i, i]
    return q, r


def perform_qr_iterations(u):
    a = u
    dimension = (u.shape[:ACCURACY])[0]
    actual_tolerance = u[0, dimension - 1]
    eigenvectors = np.identity(dimension)
    count = 0
    while actual_tolerance > TOLERANCE:
        q, r = qr_decomposition(a)
        eigenvectors = np.dot(eigenvectors, q)
        a = np.dot(r, q)
        actual_tolerance = a[0, dimension - 1]
        count += 1
    eigenvalues = np.diag(a)

    # make eigenvectors clearer to read
    for i in range(dimension):
        eigenvectors[:, i] = eigenvectors[:, i] / min(eigenvectors[:, i])
    eigenvectors = np.around(eigenvectors, decimals=ACCURACY)

    # extract eigenvectors from matrix
    eigenvectors_list = []
    for i in range(dimension):
        eigenvectors_list.append(eigenvectors[:, i].tolist())

    return eigenvalues, eigenvectors_list


def write_result(a, values, vectors):
    f = open('results.txt', 'w')
    # write input matrix a to file
    f.write("Input array:\n")
    f.write(np.array_str(a))
    f.write("\n\n")
    # write eigenvalues
    f.write("Eigenvalues of input matrix (to %iDP):\n" % ACCURACY)
    values = np.around(values, decimals=ACCURACY)
    f.write("%s\n\n" % values.tolist())
    # write eigenvectors
    f.write("Eigenvectors of input matrix (to %iDP):\n" % ACCURACY)
    f.write("%s\n" % vectors)
    f.close()


def main():
    # accuracy of output values in decimal places
    global ACCURACY
    config = input("Enter 'random' for a random matrix, or 'file' "
                   "to read the matrix from a file:\n")

    if config == 'random':
        dimension = int(input("Enter the dimension for the matrix, i.e. "
                              "enter '3' for a 3x3 matrix:\n"))
        # generate random symmetric matrix
        temp = np.random.random_sample(size=(dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                temp[j, i] = random.randint(0, MAX_VALUE) * temp[j, i]
        # ensure matrix is symmetric
        a = (temp + temp.T) / 2

    elif config == 'file':
        filename = input("Enter filename to read from:\n"
                         "Note: input must be a whitespace separated "
                         "file with spaces between columns and newline "
                         "characters between rows. See example_matrix.txt.\n")
        # use matrix loaded from file
        a = np.loadtxt(filename)
        if not (a.T == a).all():
            exit("Error: array must be symmetric.")

    else:
        exit("Error: invalid argument.")

    ACCURACY = int(input("Please enter desired accuracy for the results, "
                         "in decimal places:\n"))
    # perform qr iterations
    eigenvalues, eigenvectors = perform_qr_iterations(a)
    write_result(a, eigenvalues, eigenvectors)
    print("Results written to results.txt")

if __name__ == "__main__":
    main()
