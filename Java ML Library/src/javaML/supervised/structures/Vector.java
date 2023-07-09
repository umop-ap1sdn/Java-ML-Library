package javaML.supervised.structures;

/**
 * The Vector class is a subset of the Matrix class and consists of Matrices that have only 1 column<br>
 * Vectors will be used by Neuron layers since NLayers are meant to hold 1 Dimensional data
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class Vector extends Matrix {
	
	double[] vector;
	
	/**
	 * Basic constructor to create a new Vector from scratch<br>
	 * Behaves similar to its respective function in the Matrix class, however only requires one size
	 * parameter, since there will only be 1 column
	 * 
	 * @param size Number of rows in the Vector
	 * @param fillCode Identifier describing how to fill the vector<br>
	 * Use constants from the Matrix class to determine how the vector will be initialized
	 */
	public Vector(int size, int fillCode){
		super(size, 1, fillCode);
		this.vector = ArrayFunctions.transpose1D(matrix);
	}
	
	/**
	 * Constructor to build a vector from a pre-made array
	 * @param vector 1D array to be turned into a vector
	 */
	public Vector(double[] vector){
		//Calls the transpose function to 
		super(ArrayFunctions.transpose(vector));
		this.vector = ArrayFunctions.copyArray(vector, vector.length);
		
	}
	
	/**
	 * Constructor to build a vector from a matrix style 2D array<br>
	 * Note that this constructor will fail if the matrix has more than 1 columns
	 * @param matrix 2D array which must have 1 column
	 */
	protected Vector(double[][] matrix){
		super(matrix);
		//1D vector array is created by transposing from 2D back to 1D
		
		int rows = matrix.length;
		int cols = matrix[0].length;
		
		this.vector = ArrayFunctions.transpose1D(ArrayFunctions.copyArray(matrix, rows, cols));
		
	}
	
	/**
	 * Shortcut method for the vector class to add with 2 vectors without needing
	 * the Matrix parent method
	 * @param v1 Vector 1
	 * @param v2 Vector 2
	 * @return resultant vector of adding v1 to v2
	 */
	public static Vector add(Vector v1, Vector v2) {
		Matrix vec = Matrix.add(v1, v2);
		return new Vector(vec.matrix);
	}
	
	/**
	 * Shortcut method for the vector class to scale a vector without requiring the client to call the 
	 * Matrix parent method
	 * @param v1 Vector to be scaled
	 * @param scalar Multiplier to scale the vector by
	 * @return Resultant vector of scaling
	 */
	public static Vector scale(Vector v1, double scalar) {
		Matrix vec = Matrix.scale(v1, scalar);
		return new Vector(vec.matrix);
	}
	
	/**
	 * Function to get the 1D array form of the vector
	 * @return 1D array vector
	 */
	public double[] getVector() {
		return this.vector;
	}
	
	/**
	 * Function to set or reset the vector values
	 * @param vector 1D array to set the vector too
	 */
	public void setVector(double[] vector) {
		this.vector = ArrayFunctions.copyArray(vector, vector.length);
		//Change to 2D array for parent class
		super.setMatrix(ArrayFunctions.transpose(vector));
	}
	
	/**
	 * Function to get a particular value from the vector 
	 * @param row Location of the vector to get data from
	 * @return Value at the specified location
	 */
	public double getValue(int row) {
		return super.getValue(row, 0);
	}
	
	/**
	 * Function to set a value at a particular location
	 * @param value Value to be set
	 * @param row Location to set the value
	 */
	public void setValue(double value, int row) {
		super.setValue(value, row, 0);
		this.vector[row] = value;
	}
	
	@Override
	public void simplePrint() {
		for(int row = 0; row < rows; row++) {
			System.out.printf("[%.2f]%n", vector[row]);
		}
		
		System.out.println();
	}
}
