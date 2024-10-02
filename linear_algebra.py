# Version 2

class Matrix:
    "Important: When adding to / changing a Matrix and you input lists, remember to put them as [[1, 2], [3, 4]...]"

    def __init__(self, lists, output = list):
        length = len(lists[0])
        if not all(len(r) == length for r in lists):
            raise IndexError("Rows must be the same length")
        if output not in (list, Matrix):
            raise TypeError("Output must be a list or Matrix type")
        self.lists = lists
        self.output = output

    def __repr__(self):
        return "\n".join(repr(s) for s in self.lists)

    @classmethod
    def new(cls, rows, columns, value = 0, output = list):
        "Make a new Matrix with the same value (default 0)"
        return cls([[value] * columns] * rows, output = output)

    @classmethod
    def vector(cls, values, horizontal = True, output = list):
        "Make a new vector (Matrix with 1 dimension) from a single unnested list"
        return cls(([values] if horizontal else [[v] for v in values]), output = output)

    @classmethod
    def new(cls, rows, columns, value = 0, output = list):
        "Return a Matrix with all values 0 (or whatever you input)"
        return cls([[value for x in range(columns)] for x in range(rows)], output = output)

    @classmethod
    def identity(cls, length, output = list):
        "Return an identity Matrix"
        m = cls([[0 for x in range(length)] for x in range(length)], output = output)
        for x in range(length):
            m[x, x] = 1
        return m

    def columns(self):
        "Return the number of columns that this matrix has"
        return len(self.lists[0])

    height = columns

    def rows(self):
        "Return the number of rows that this matrix has"
        return len(self.lists)
    
    width = rows

    def copy(self, output = None):
        "Make a copy of the matrix, always returned as a Matrix object"
        # Copy deep, copy each row and copy each column (but not the elements within each column)
        return Matrix([[v for v in r] for r in self.lists], output = Matrix if output is None else output)

    def __getitem__(self, args):
        "You can either get a row or a specific value (row, column)"
        if isinstance(args, tuple): # Get one value
            return self.lists[args[0]][args[1]]
        if isinstance(args, slice): # Doing something like matrix[:] breaks the return value
            raise IndexError("Must do matrix[n] to get row n or matrix[a, b] to get row a column b")
        return Matrix([self.lists[args]]) if self.output == Matrix else self.lists[args]

    def get_row(self, row):
        "Return row n of the matrix. n = 0 returns the first row"
        return self[row]

    def get_column(self, column):
        "Return column n of the matrix. n = 0 returns the first column"
        return [r[column] for r in self.lists]

    def get_region(self, r_up, r_down, c_left, c_right):
        """Obtain an entire region in a Matrix.
        Important: This works like slices, so the r_down and c_right values are ignored.
        So Matrix.set_region(r_up = 1, r_down = 3...) can only obtains values in the 2nd and 3rd rows"""
        if r_down > self.rows() or c_right > self.columns():
            raise IndexError("Index out of range")
        if r_down <= r_up or c_right <= c_left:
            raise IndexError("Invalid coordinates (less than 1 in width)")
        rows = r_down - r_up
        columns = c_right - c_left
        if rows == columns == 1: # Just one coordinate selected
            return self.__getitem__((r_up, c_left))
        # Now we can have (part of) a row, a column, or a 2-dimensional region selected
        if rows == 1 and columns != 1: # Part of a row
            r = self.lists[r_up][c_left : c_right]
            return Matrix([r]) if self.output == Matrix else r
        elif rows != 1 and columns == 1: # Part of a column
            r = [row[c_left] for row in self.lists[r_up : r_down]]
            return Matrix([[v] for v in r]) if self.output == Matrix else r
        else:
            r = [row[c_left : c_right] for row in self.lists[r_up : r_down]]
            return Matrix(r) if self.output == Matrix else r

    def __setitem__(self, args, value):
        "You can either set a row or a specific value (row, column)"
        if isinstance(args, tuple): # Changing one value
            self.lists[args[0]][args[1]] = value
        else: # Replacing a row
            if not isinstance(value, (list, Matrix)):
                raise TypeError("Row must be a list or another Matrix")
            if isinstance(value, Matrix):
                if value.rows() > 1:
                    raise ValueError("Must be a horizontal Matrix with 1 row")
                value = value[0]
            if len(value) != len(self.lists[0]):
                raise IndexError("Rows must be the same length")
            self.lists[args] = value

    def set_region(self, r_up, r_down, c_left, c_right, values):
        """Obtain an entire region in a Matrix.
        Important: This works like slices, so the r_down and c_right values are ignored.
        So Matrix.set_region(r_up = 1, r_down = 3...) can only obtains values in the 2nd and 3rd rows
        Also important: If lists are inputted, they must be nested like [[1, 2, 3], [4, 5, 6]]"""
        if r_down > self.rows() or c_right > self.columns():
            raise IndexError("Index out of range")
        if r_down <= r_up or c_right <= c_left:
            raise IndexError("Invalid coordinates (less than 1 in width)")
        rows = r_down - r_up
        columns = c_right - c_left
        if rows == columns == 1: # Just one coordinate selected
            return self.__setitem__((r_up, c_left), values)
        # Now we can have (part of) a row, a column, or a 2-dimensional region selected
        # First, we must check that the values match the correct dimensions
        if isinstance(values, Matrix):
            values = values.lists
        if not isinstance(values, list):
            raise TypeError("Row must be a list or another Matrix")
        if len(values[0]) != columns or len(values) != rows:
            raise IndexError("Incorrect row and/or column lengths")
        for row, value_row in zip(self.lists[r_up : r_down], values):
            row[c_left:c_right] = value_row

    def __delitem__(self, index):
        "Delete rows of the Matrix (ignores the lower end)"
        del self.lists[index]

    def delete_rows(self, upper, lower = None):
        "Delete rows of the Matrix (ignores the lower end)"
        if lower is None:
            lower = upper + 1
        del self.lists[upper : lower]

    def delete_columns(self, left, right):
        "Delete columns of a Matrix (ignores the rightmost)"
        for r in self.lists:
            del r[left : right]

    remove_columns = delete_columns

    def extend_rows(self, values):
        """Add rows to the bottom of the Matrix
        If lists are inputted, they must be nested like [[1, 2, 3], [4, 5, 6]]"""
        if isinstance(values, Matrix):
            values = values.lists
        if not isinstance(values, list):
            raise TypeError("Values must be a list or another Matrix")
        columns = self.columns()
        if not all(len(r) == columns for r in values):
            raise IndexError("Rows must be the same length")
        self.lists.extend([r[:] for r in values])

    def extend_columns(self, values):
        """Add rows to the bottom of the Matrix
        If lists are inputted, they must be nested like [[1, 2, 3], [4, 5, 6]]"""
        if isinstance(values, Matrix):
            values = values.lists
        if not isinstance(values, list):
            raise TypeError("Values must be a list or another Matrix")
        if len(values) != self.rows():
            raise IndexError("Columns must be the same length")
        values_row_length = len(values[0])
        if any(len(r) != values_row_length for r in values):
            raise IndexError("Columns must be the same length")
        for r, v in zip(self.lists, values):
            r.extend(v)

    def insert(self, index, values):
        """Insert a row or column to the Matrix in front of the index specified
        If lists are inputted, they must be nested like [[1, 2, 3], [4, 5, 6]]"""
        if isinstance(values, Matrix):
            if not values.isvector():
                raise ValueError("Matrix must be a vector")
            values = values.lists
        if not isinstance(values, list):
            raise TypeError("Values must be a list or another Matrix")
        if len(values) == 1 and len(values[0]) != 1: # Horizontal vector
            if len(values[0]) != self.columns():
                raise IndexError("Rows must be the same length")
            self.lists.insert(index, values[0])
        if len(values) != 1 and len(values[0]) == 1: # Vertical vector
            if len(values) != self.rows():
                raise IndexError("Columns must be the same length")
            for r, v in zip(self.lists, values):
                r.insert(index, v[0])

    def __eq__(self, m):
        if isinstance(m, Matrix):
            m = m.lists
        if hasattr(m, "__iter__") and hasattr(m, "__len__") and self.rows() == len(m) and self.columns() == len(m[0]) \
           and all(all(v1 == v2 for (v1, v2) in zip(r1, r2)) for (r1, r2) in zip(self.lists, m)):
            return True
        return False

    def __add__(self, m):
        "Add 2 matrices"
        if m == 0:
            return self # This is necessary because when sum() is used, it adds the first value to zero
        if isinstance(m, Matrix):
            m = m.lists
        if self.rows() != len(m) or self.columns() != len(m[0]):
            raise IndexError("Rows and columns must be the same length")
        return Matrix([[v1 + v2 for (v1, v2) in zip(r1, r2)] for (r1, r2) in zip(self.lists, m)], output = self.output)

    __radd__ = __add__

    def add_all(self, num):
        "Create a new Matrix which is a copy of this Matrix with a number num added to each element"
        "Note that Matrix.__add__() does not do this by default"
        return Matrix([[v + num for v in r] for r in self.lists], output = self.output)

    def __sub__(self, m):
        "Subtract 2 matrices"
        if m == 0:
            return self # This is necessary
        if isinstance(m, Matrix):
            m = m.lists
        if self.rows() != len(m) or self.columns() != len(m[0]):
            raise IndexError("Rows and columns must be the same length")
        return Matrix([[v1 - v2 for (v1, v2) in zip(r1, r2)] for (r1, r2) in zip(self.lists, m)], output = self.output)

    def __rsub__(self, m):
        "Subtract 2 matrices"
        if isinstance(m, Matrix):
            m = m.lists
        if self.rows() != len(m) or self.columns() != len(m[0]):
            raise IndexError("Rows and columns must be the same length")
        return Matrix([[v2 - v1 for (v1, v2) in zip(r1, r2)] for (r1, r2) in zip(self.lists, m)], output = self.output)

    def __mul__(self, value):
        "Multiply a Matrix. Can be scalar or by another Matrix"
        if isinstance(value, Matrix):
            if self.isvector() and value.isvector():
                return self.vectors_mul(value)
            if self.columns() == value.rows():
                return self.mul_rows(value)
            if self.rows() == value.columns():
                return self.mul_columns(value)
            raise IndexError("Cannot multiply these matrices")
        return Matrix([[v * value for v in row] for row in self.lists], output = self.output) # Scalar

    __rmul__ = __mul__

    def mul_rows(self, matrix_b):
        "C = AB. Multiply Matrix B's columns by the rows of self"
        if self.columns() == matrix_b.rows():
            return Matrix([[sum(v * c for v, c in zip(row, col)) for col in matrix_b.iterate_columns()] for row in self.lists])
        raise IndexError("The first Matrix has {} columns while the second has {} rows".format(self.columns(), matrix_b.rows()))

    def mul_columns(self, matrix_b):
        "C = AB. Multiply Matrix B's rows by the columns of self. (This is not the convention of bA notation, where vector b is placed in front)"
        if self.rows() == matrix_b.columns():
            return Matrix([[sum(v * c for v, c in zip(row, col)) for col in self.iterate_columns()] for row in matrix_b.lists])
        raise IndexError("The first Matrix has {} rows while the second has {} columns".format(self.rows(), matrix_b.columns()))

    def __pow__(self, n):
        "Multiply a Matrix by itself n times"
        for x in range(n - 1):
            self *= self
        return self

    def vectors_mul(self, vector_b):
        "Multiply 2 vectors for a scalar product"
        if not self.isvector() or not vector_b.isvector():
            raise IndexError("Both Matrices must be vectors")
        vector_a = self.lists[0] if self.rows() == 1 else [v[0] for v in self.lists]
        vector_b = vector_b.lists[0] if vector_b.rows() == 1 else [v[0] for v in vector_b.lists]
        if len(vector_a) == len(vector_b):
            return sum(a * b for a, b in zip(vector_a, vector_b))
        raise IndexError("Vectors must be the same length")

    def isvector(self):
        "Returns if a Matrix is a vector (1 dimensional)"
        return (self.rows() == 1 and self.columns() != 1) or (self.rows != 1 and self.columns() == 1)

    def flatten(self):
        "Yield the values of the matrix like a book"
        for r in self.lists:
            yield from r

    def iterate_rows(self):
        "Iterate the rows"
        return self.lists.__iter__()

    def iterate_columns(self):
        "Iterate the columns, from left to right"
        for n in range(len(self.lists[0])):
            yield [row[n] for row in self.lists]

    def transpose(self):
        "Transpose a Matrix"
        return Matrix([c for c in self.iterate_columns()], output = self.output)

    def row_reduce_self(self):
        """Perform row reduction on a Matrix by itself
        Important: When doing M.row_reduce(), Matrix M is not an augmented matrix."""
        self_copy = self.copy(output = Matrix)
        lists = self_copy.lists
        columns = len(lists[0])
        def leftmost_index(row):
            for index, value in enumerate(row):
                if value != 0:
                    return index
            return columns # The row has only zeroes
        for counter in range(columns - 1):
            rows = iter(row for row in lists if leftmost_index(row) == counter)
            try:
                pivot = next(rows) # Now that we have rows with the same leftmost, add / subtract rows if they share the same leftmost column
                pivot_first = pivot[counter]
            except StopIteration: # Missing pivot or zeroes row
                continue
            for row in rows: # Reduce any other rows with the same leftmost nonzero
                scale_factor = row[counter] / pivot_first
                row[:] = [r - p * scale_factor for p, r in zip(pivot, row)]
        # The Matrix may need some row swaps
        lists.sort(key = lambda r: leftmost_index(r))
        return self_copy

    def row_echelon_form(self, constants = None, merge = False):
        """Perform row reduction on a Matrix
        Important: When doing M.row_reduce(), Matrix M is not an augmented matrix.
        If argument merge == True, then return an augmented Matrix with any constants Matrix attached to the coefficient Matrix"""
        if constants is None:
            return self.row_reduce_self()
        if not isinstance(constants, Matrix):
            raise TypeError("Expected a Matrix")
        if self.rows() != constants.rows():
            raise TypeError("The constants Matrix must have the same number of rows")
        self_copy = self.copy()
        constants_copy = constants.copy()
        lists = self_copy.lists
        columns = len(lists[0]) # Only for the coefficient Matrix
        augmented = list(zip(lists, constants_copy.lists)) # Here is our augmented Matrix
        def leftmost_index(row):
            for index, value in enumerate(row):
                if value != 0:
                    return index
            return columns # The coefficient Matrix row has only zeroes
        for counter in range(columns - 1):
            rows = iter(row for row in augmented if leftmost_index(row[0]) == counter) # Includes both the coefficient and constants Matrices
            try:
                pivot = next(rows) # Now that we have rows with the same leftmost, add / subtract rows if they share the same leftmost column
                pivot_first = pivot[0][counter]
            except StopIteration: # Missing pivot or zeroes row
                continue
            for coefficients, constants in rows:
                scale_factor = coefficients[counter] / pivot_first
                coefficients[:] = [r - p * scale_factor for p, r in zip(pivot[0], coefficients)]
                constants[:] = [r - p * scale_factor for p, r in zip(pivot[1], constants)]
        # The Matrix may need some row swaps
        augmented.sort(key = lambda r: leftmost_index(r[0]))
        if merge: # Now return
            return Matrix([coefficient + constants for (coefficient, constants) in augmented], output = self.output)
        return Matrix([r[0] for r in augmented], output = self.output), Matrix([r[1] for r in augmented], output = self.output)

    reduce = row_reduce = ref = row_echelon_form

    def backsub(self, constants, merge = False):
        "Perform backsubstitution on a Matrix in upper triangle form"
        # Important: The Matrix MUST be in upper triangular form
        if not isinstance(constants, Matrix):
            raise TypeError("Expected a Matrix")
        if self.rows() != constants.rows():
            raise TypeError("The constants Matrix must have the same number of rows")
        self_copy = self.copy(output = Matrix)
        constants_copy = constants.copy(output = Matrix)
        augmented = list(zip(self_copy.lists, constants_copy.lists))
        columns = len(self_copy.lists[0])
        for n, (coefficient_row, constants_row) in enumerate(reversed(augmented), 1):
            target = coefficient_row[-n]
            if target == 0 or any(n != 0 for n in coefficient_row[:-n]):
                raise ValueError("Matrix not in upper triangular form")
            nonzero_positions = [n for n, x in enumerate(coefficient_row) if x != 0]
            if nonzero_positions:
                for coeff_row, const_row in augmented:
                    if coefficient_row is not coeff_row: # Do not subtract the same row from itself
                        coeff_row_nonzeros = [n for n, x in enumerate(coeff_row) if x != 0]
                        if coeff_row_nonzeros and all(z in nonzero_positions for z in coeff_row_nonzeros):
                            multiply_factor = coefficient_row[coeff_row_nonzeros[0]] / coeff_row[coeff_row_nonzeros[0]]
                            coefficient_row[:] = [a - b for a, b in zip(coefficient_row, (n * multiply_factor for n in coeff_row))]
                            constants_row[:] = [a - b for a, b in zip(constants_row, (n * multiply_factor for n in const_row))]
            constants_row[:] = [n / target for n in constants_row]
            coefficient_row[-n] = 1
        if merge:
            self_copy.extend_columns(constants_copy)
            return self_copy
        return self_copy, constants_copy

    def rref(self, constants, merge = False):
        """Turn a Matrix into reduced row echelon (rref) form
        Important: When doing M.row_reduce(), Matrix M is not an augmented matrix.
        If argument merge == True, then return an augmented Matrix with any constants Matrix attached to the coefficient Matrix"""
        if not isinstance(constants, Matrix):
            raise TypeError("Expected a Matrix")
        if self.rows() != constants.rows():
            raise TypeError("The constants Matrix must have the same number of rows")
        a, b = self.row_reduce(constants)
        a, b = a.backsub(b)
        if merge:
            a.extend_columns(b)
            return a
        return a, b

    def inverse(self):
        "Return the inverse if a Matrix, if it's nonsingular"
        left, right = self.row_reduce(self.identity(len(self.lists)), merge = False)
        try:
            return left.backsub(right)[1]
        except ValueError:
            return None

    def elimination_matrix(self): # May need to fix because not all matrices can have upper triangular form
        "Return the elimination Matrix which converts A into upper triangular form"
        # MA = U -> M = U * A-1
        return self.row_reduce(self.identity(len(self.lists)), merge = False)[1]

    def elementary_breakdown(self):
        "Yield the steps of row reducing a Matrix along with the elementary Matrices"
        # The last tuple yielded is the reduced Matrix along with the elementary Matrix dealing with row swaps
        # If the last elementary Matrix is an identity Matrix, no row swaps occurred
        self_copy = self.copy(output = Matrix)
        lists = self_copy.lists
        columns = len(lists[0])
        matrix_rows = len(lists)
        def leftmost_index(row):
            for index, value in enumerate(row):
                if value != 0:
                    return index
                if index == columns:
                    return columns # The row has only zeroes
        for counter in range(columns - 1):
            rows = iter((row, n) for n, row in enumerate(lists) if leftmost_index(row) == counter)
            try:
                pivot = next(rows)[0] # Now that we have rows with the same leftmost, add / subtract rows if they share the same leftmost column
                pivot_first = pivot[counter]
            except StopIteration: # Missing pivot or zeroes row
                continue
            for row, n in rows: # Reduce any other rows with the same leftmost nonzero
                scale_factor = row[counter] / pivot_first
                row[:] = [r - p * scale_factor for p, r in zip(pivot, row)]
                elementary = Matrix.identity(matrix_rows)
                elementary[n, counter] = -scale_factor
                yield self_copy, elementary
        # The Matrix may need some row swaps
        identity = Matrix.identity(matrix_rows)
        self_copy.extend_columns(identity)
        lists.sort(key = lambda r: leftmost_index(r))
        row_swaps = self_copy.get_region(0, len(lists), columns, columns + matrix_rows)
        for r in lists:
            r = r[:columns]
        yield self_copy.get_region(0, len(lists), 0, columns), row_swaps

    def lu_decomposition(self):
        "Return L, U and P (in that order) for PA = LU in the LU decomposition of a Matrix"
        upper_triangle, row_swaps = tuple(self.elementary_breakdown())[-1]
        self_copy = row_swaps * self.copy(output = Matrix)
        lists = self_copy.lists
        columns = len(lists[0])
        lower_triangle = Matrix.identity(len(lists))
        def leftmost_index(row):
            for index, value in enumerate(row):
                if value != 0:
                    return index
            return columns # The row has only zeroes
        for counter in range(columns - 1):
            rows = iter((row, n) for n, row in enumerate(lists) if leftmost_index(row) == counter)
            try:
                pivot = next(rows)[0] # Now that we have rows with the same leftmost, add / subtract rows if they share the same leftmost column
                pivot_first = pivot[counter]
            except StopIteration: # Missing pivot or zeroes row
                continue
            for row, n in rows: # Reduce any other rows with the same leftmost nonzero
                scale_factor = row[counter] / pivot_first
                row[:] = [r - p * scale_factor for p, r in zip(pivot, row)]
                lower_triangle[n, counter] = scale_factor
        return lower_triangle, upper_triangle, row_swaps

    lu = lu_decomposition

if __name__ == "__main__": # Test cases
    basic = Matrix([[1, 1, -1], [1, 3, 1], [2, 3, 3]])
    basic_constants = Matrix.vector([1, 1, -2], False)
    row_swaps = Matrix([[1, 1, -1], [1, 1, 1], [1, 2, 2]])
    row_swaps_constants = Matrix.vector([1, 2, 1], False)
    no_solution = Matrix([[1, 1, -1], [1, 2, 2], [2, 3, 1]])
    no_solution_constants = Matrix.vector([0, 1, -2], False)
    many_solutions = Matrix([[1, 1, -1], [2, 1,1], [1, 0, 2]])
    many_solutions_constants = Matrix.vector([0, 1, 1], False)
    many_solutions2 = Matrix([[1, 1, 1, 1], [1, 2, 3, 4], [2, 2, 3, 3], [2, 1, 2, 1]])
    many_solutions2_constants = Matrix.vector([0, 0, 0, 0], False)
    missing_pivot = Matrix([[1, 1, -1], [1, 1, 1], [1, 1, -2]])
    missing_pivot_constants = Matrix.vector([2, 0, 3], False)
    testthis = Matrix([[1, 2, 3, 4], [0, 1, 0, 1], [2, 3, 5, 7]])
    testthis_constants = Matrix.vector([3, -4, 8], False)
