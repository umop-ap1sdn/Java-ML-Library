package javaML.supervised.structures;

/**
 * Math heavy class used to model a matrix<br>
 * This class will be used primarily for ConnectionLayers which can be modeled
 * as matrices<br><br>
 * 
 * Capable of performing Matrix addition, multiplication, point-wise multiplication and scaling<br>
 * 4 options for Matrix initialization, including all 0s, all 1s, random filled, and identity matrix
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class Matrix {
	
	/**
	 * Valid fill codes for matrix initialization<br>
	 * <ul>
	 * <li>FILL_ZERO initializes every element to 0</li>
	 * <li>FILL_ONE initializes every element to 1</li>
	 * <li>FILL_RANDOM randomizes every element</li>
	 * <li>FILL_IDENTITY fills the long diagonal from top left to bottom right with 1</li>
	 * </ul>
	 */
	public static final int FILL_ZERO = 1;
	public static final int FILL_ONE = 2;
	public static final int FILL_RANDOM = 3;
	public static final int FILL_IDENTITY = 4;
	
	protected int rows, columns;
	protected double[][] matrix;
	
	/**
	 * Basic constructor to create a new matrix from scratch
	 * @param rows Number of rows in the matrix
	 * @param columns Number of columns in the matrix
	 * @param fillCode Code for how to initialize the matrix<br>
	 * Use constants from the Matrix (this) class to choose how to initialize the matrix
	 */
	public Matrix(int rows, int columns, int fillCode){
		this.rows = rows;
		this.columns = columns;
		
		this.matrix = new double[rows][columns];
		
		this.initialize(fillCode);
		
	}
	
	/**
	 * Basic constructor to create an identity matrix<br>
	 * The identity matrix is a matrix filled with 0s except for the diagonal which is filled with 1s<br>
	 * The purpose of this matrix is to be multiplied of depth equal to the matrix size<br>
	 * The result will be a vector equal to the original vector
	 * @param size Size of a matrix to create, which will be a square (rows = columns = length)
	 */
	public Matrix(int size) {
		this.rows = size;
		this.columns = size;
		
		this.matrix = new double[rows][columns];
		this.initialize(FILL_IDENTITY);
	}
	
	/**
	 * Advanced constructor to turn an already existing 2D array into a matrix data type
	 * @param matrix 2D array to be inserted in the Matrix data type
	 */
	public Matrix(double[][] matrix){
		this.rows = matrix.length;
		this.columns = matrix[0].length;
		
		this.matrix = ArrayFunctions.copyArray(matrix, rows, columns);
	}
	
	/**
	 * Initialize function to be called by the basic constructors
	 * @param fillCode User inputed fill code that describes how to build the matrix
	 */
	private void initialize(int fillCode) {
		switch(fillCode) {
		case FILL_ZERO:
			fill0();
			break;
		case FILL_ONE:
			fill1();
			break;
		case FILL_RANDOM:
			fillRandom();
			break;
		case FILL_IDENTITY:
			fillIden();
			break;
		}
	}
	
	/**
	 * Function for the FILL_ZERO fill code<br>
	 * As the name suggests, the matrix is filled with 0s
	 */
	private void fill0() {
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < columns; col++) {
				matrix[row][col] = 0;
			}
		}
	}
	
	/**
	 * Function for the FILL_ONE fill code<br>
	 * As the name suggests, the matrix is filled with 1s
	 */
	private void fill1() {
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < columns; col++) {
				matrix[row][col] = 1;
			}
		}
	}
	
	/**
	 * Function for the FILL_RANDOM fill code<br>
	 * As the name suggests, the matrix is filled with random values
	 */
	private void fillRandom() {
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < columns; col++) {
				matrix[row][col] = (Math.random() * 2) - 1;
			}
		}
	}
	
	/**
	 * Function for the FILL_IDENTITY fill code<br>
	 * The matrix will be built as an identity matrix
	 */
	private void fillIden() {
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < columns; col++) {
				if(row == col) matrix[row][col] = 1;
				else matrix[row][col] = 0;
			}
		}
	}
	
	/**
	 * Function to add 2 Matrices together<br>
	 * Matrix addition compared to other matrix math is rather simple<br>
	 * The resultant matrix is calculated by computing the addition of the value in matrix 1 and
	 * matrix 2 at every coordinate point
	 * @param m1 First matrix
	 * @param m2 Second matrix
	 * @return Returns the addition of the 2 matrices<br>
	 * If matrices are incompatible (Matrices do not have the same size), returns null
	 */
	public static Matrix add(Matrix m1, Matrix m2) {
		if(m1.rows != m2.rows && m1.columns != m2.columns) return null;
		
		double[][] mat = new double[m1.rows][m1.columns];
		
		for(int row = 0; row < m1.rows; row++) {
			for(int col = 0; col < m1.columns; col++) {
				//Perform piece-wise multiplication on each element of the 2 matrices
				mat[row][col] = m1.getValue(row, col) + m2.getValue(row, col);
			}
		}
		
		return new Matrix(mat);
	}
	
	/**
	 * Function to scale an entire matrix by a constant<br>
	 * Every element of the matrix will be multiplied by this constant
	 * @param m1 Matrix
	 * @param scalar scaling constant
	 * @return Returns the scaled matrix
	 */
	public static Matrix scale(Matrix m1, double scalar) {
		double[][] mat = new double[m1.rows][m1.columns];
		
		for(int row = 0; row < m1.rows; row++) {
			for(int col = 0; col < m1.columns; col++) {
				//Multiply every coordinate by the scalar
				mat[row][col] = m1.getValue(row, col) * scalar;
			}
		}
		
		return new Matrix(mat);
	}
	
	/**
	 * This function performs a point-wise Matrix multiplication<br>
	 * This process is similar to Matrix addition where every element of the new matrix[i,j]
	 * will be equal to m1[i,j] * m2[i,j]
	 * @param m1 First matrix
	 * @param m2 Second matrix
	 * @return Returns the resultant matrix of m1 point-multiplied by m2<br>
	 * If the matrices are not of equal size, they are not compatible and will return null
	 */
	public static Matrix linearMultiply(Matrix m1, Matrix m2) {
		if(m1.rows != m2.rows || m1.columns != m2.columns) return null;
		
		double[][] matrix = new double[m1.rows][m1.columns];
		
		for(int row = 0; row < m1.rows; row++) {
			for(int col = 0; col < m1.columns; col++) {
				//Perform piece-wise multiplication on the 2 matrices
				matrix[row][col] = m1.getValue(row, col) * m2.getValue(row, col);
			}
		}
		
		return new Matrix(matrix);
	}
	
	/**
	 * Function to transpose a Matrix
	 * @param m1 Matrix to be transposed
	 * @return Returns the transposed matrix
	 */
	public static Matrix transpose(Matrix m1) {
		double[][] mat = ArrayFunctions.transpose(m1.matrix);
		return new Matrix(mat);
	}
	
	/**
	 * Function to perform Matrix multiplication<br>
	 * Matrix multiplication is much more complex than addition/pointwise multiplication.
	 * A point the resultant matrix[i,j] = m1[i, 0-n] * m2[0-n, j].
	 * The resultant matrix has the dimensions equal to m = rows in m1, n = columns in m2,
	 * while the columns in m1 must be equal to the rows in m2 in order to be compatible<br><br>
	 * If the 2 matrices are incompatible, one check is made to see if they would be compatible the
	 * other way around (since matrices are not commutable, m1 * m2 != m2 * m1)
	 * @param m1 First matrix
	 * @param m2 Second matrix
	 * @return The resultant matrix of m1 multiplied by m2<br>
	 * If the matrices are incompatible, and the reverse compatibility check fails, returns null
	 */
	public static Matrix multiply(Matrix m1, Matrix m2) {
		if(m1.columns != m2.rows) {
			if(m2.columns != m1.rows) return null;
			else return multiply(m2, m1);
		}
		
		double[][] matrix = new double[m1.rows][m2.columns];
		
		for(int row = 0; row < m1.rows; row++) {
			for(int col = 0; col < m2.columns; col++) {
				//Call helper function to get the result for every coordinate
				matrix[row][col] = multiplyCoord(m1, m2, row, col);
			}
		}
		
		return new Matrix(matrix);
	}
	
	/**
	 * Helper Function to be called only by the multiply function<br>
	 * Takes in the 2 matrices being currently multiplied and a coordinate point for which to multiply
	 * around<br>
	 * This method will be called for every coordinate point in the resultant matrix
	 * @param m1 Matrix 1
	 * @param m2 Matrix 2
	 * @param row Row of multiplication
	 * @param col Column of multiplication
	 * @return the result of multiplication at the particular coordinate
	 */
	private static double multiplyCoord(Matrix m1, Matrix m2, int row, int col) {
		
		double sum = 0;
		int length = m1.columns;
		
		//The result is equal to the sum of 
		//every element on a particular row of m1 * its respective element in the particular column of m2 
		for(int index = 0; index < length; index++) {
			sum += m1.getValue(row, index) * m2.getValue(index, col);
		}
		
		return sum;
	}
	
	
	/**
	 * Getter for the array for the matrix
	 * @return Matrix as a 2D array
	 */
	public double[][] getMatrix(){
		return this.matrix;
	}
	
	/**
	 * Function to set the matrix to a new 2D array<br>
	 * If the array does not have the expected size of the matrix, function does nothing
	 * @param matrix new 2D array matrix
	 */
	public void setMatrix(double[][] matrix) {
		if(matrix.length != rows || matrix[0].length != 0) return; 
		
		this.matrix = ArrayFunctions.copyArray(matrix, rows, columns);
	}
	
	/**
	 * Function to set the value of a particular coordinate
	 * @param value
	 * @param row 
	 * @param col
	 */
	public void setValue(double value, int row, int col) {
		this.matrix[row][col] = value;
	}
	
	/**
	 * Function to get the value of a particular coordinate
	 * @param row
	 * @param col
	 * @return Value of matrix[row][column]
	 */
	public double getValue(int row, int col) {
		return matrix[row][col];
	}
	
	/**
	 * Function to get the Matrix as a vector
	 * Must have exactly 1 column to be eligible to be returned as a Vector
	 * @return Returns the matrix as a vector data type
	 */
	public Vector getAsVector() {
		if(this.columns != 1) return null;
		return new Vector(this.matrix);
	}
	
	/**
	 * Getter for the number of columns in the Matrix
	 * @return Columns
	 */
	public int getColumns() {
		return this.columns;
	}
	
	/**
	 * Getter for number of rows in the matrix
	 * @return Rows
	 */
	public int getRows() {
		return this.rows;
	}
	
	/**
	 * Quick method to print the values of the array in a readable format.<br>
	 * Rounds each value to 4 decimal places, starts and ends each lines with [ and ] respectively.
	 */
	public void simplePrint() {
		for(int row = 0; row < rows; row++) {
			System.out.print("[");
			for(int col = 0; col < columns; col++) {
				System.out.printf("%.4f", matrix[row][col]);
				
				if(col < columns - 1) System.out.print(", ");
			}
			
			System.out.println("]");
		}
		
		System.out.println();
	}
	
	@Override
	public String toString() {
		String ret = "";
		
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < columns; col++) {
				ret += matrix[row][col];
				if(col < columns - 1) ret += ",";
			}
			
			if(row < rows - 1) ret += "\n";
		}
		
		return ret;
	}
}
