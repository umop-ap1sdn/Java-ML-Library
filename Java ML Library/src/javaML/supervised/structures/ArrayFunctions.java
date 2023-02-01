package javaML.supervised.structures;

/**
 * Basic file consisting only of static functions to manipulate arrays
 * @author Caleb Devon<br>
 * Created 10/14/2022
 *
 */
public class ArrayFunctions {
	
	/**
	 * Basic Transpose method<br>
	 * Takes in a matrix and outputs a transposed version of the matrix<br>
	 * Transposition involves turning the columns to rows and rows to columns
	 * @param matrix Matrix array to be transposed
	 * @return Transposed matrix array
	 */
	static double[][] transpose(double[][] matrix){
		int rows = matrix.length;
		int columns = matrix[0].length;
		
		double[][] ret = new double[columns][rows];
		
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < columns; col++) {
				//Swap rows and columns
				ret[col][row] = matrix[row][col];
			}
		}
		
		return ret;
	}
	
	/**
	 * Inclusive transpose overload method to allow for transposition of 1D arrays
	 * @param vector 1 Dimensional array to be transposed
	 * @return 2 Dimensional transposed array
	 */
	static double[][] transpose(double[] vector){
		int rows = vector.length;
		
		//Turn to 2D array and transpose
		double[][] ret = new double[1][rows];
		ret[0] = vector;
		return transpose(ret);
	}
	
	/**
	 * Inclusive transpose function for turning 2 Dimensional arrays into a transposed 1 Dimensional Array
	 * @param matrix Matrix to be transposed into a 1 dimensional array
	 * @return 1D array of transposed matrix <br>
	 * Will return null if matrix has more than one column (incompatible with 1D array)
	 */
	protected static double[] transpose1D(double[][] matrix) {
		//Returns null if incompatible
		if(matrix[0].length != 1) return null;
		
		int rows = matrix.length;
		
		//Create a 2D array with only 1 column
		//Effectively still 1 dimensional just expressed within 2 dimensions
		double[] ret = new double[rows];
		for(int row = 0; row < rows; row++) {
			ret[row] = matrix[row][0];
		}
		
		return ret;
	}
	
	/**
	 * Basic array function to create a deep copy of an array to ensure all data is unique
	 * @param Matrix 2D array to copy from
	 * @return Deep copied array of [matrix]
	 */
	protected static double[][] copyArray(double[][] matrix, int rows, int columns){
		double[][] ret = new double[rows][columns];
		
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < columns; col++) {
				ret[row][col] = matrix[row][col];
			}
		}
		
		return ret;
	}
}
