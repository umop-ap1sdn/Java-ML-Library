package javaML;

/**
 * The DataTransformations class is a class consisting only of static functions meant to modify a given dataset.<br>
 * Dataset modifications are commonly used in Data Science and Statistics to make data more readable, and is
 * especially effective for training learning models.
 * 
 * @author Caleb Devon
 * Added on 2/6/2023
 *
 */

public final class DataTransformations {
	
	/**
	 * Simple function to create a deep copy of a 3 dimensional double array
	 * @param data 3D array to be copied
	 * @return Copied 3D array that can be modified without affecting the original
	 */
	public static double[][][] deepCopy(double[][][] data){
		double[][][] dataCopy = new double[data.length][2][];
		
		for(int x = 0; x < data.length; x++) {
			for(int y = 0; y < data[x].length; y++) {
				dataCopy[x][y] = new double[data[x][y].length];
				for(int z = 0; z < data[x][y].length; z++) {
					dataCopy[x][y][z] = data[x][y][z];
				}
			}
		}
		
		return dataCopy;
	}
	
	/**
	 * Function to normalize a dataset to a specified range
	 * @param dataset 3D array to be normalized
	 * @param min Desired minimum
	 * @param max Desired maximum
	 * @param deepCopy Set to true to deep copy the dataset array
	 * @return returns the normalized dataset array
	 */
	public static double[][][] normalize(double[][][] dataset, double min, double max, boolean deepCopy){
		
		if(deepCopy) dataset = deepCopy(dataset);
		
		//Create vectors for the minimum and maximum values at each index in the inputs and targets data
		double[] inputMax = new double[dataset[0][0].length];
		double[] inputMin = new double[dataset[0][0].length];
		
		double[] outputMax = new double[dataset[0][1].length];
		double[] outputMin = new double[dataset[0][1].length];
		
		//Initialize arrays to the first datapoint of the dataset
		for(int index = 0; index < inputMax.length; index++) {
			inputMax[index] = dataset[0][0][index];
			inputMin[index] = dataset[0][0][index];
		}
		
		for(int index = 0; index < outputMax.length; index++) {
			outputMax[index] = dataset[0][1][index];
			outputMin[index] = dataset[0][1][index];
		}
		
		//Search whole array to find the maximums and minimums of each column
		//This is done by individual columns to acknowledge the fact that any 2 given inputs/outputs may
		//represent entirely different values and should be treated as unrelated
		for(int index = 1; index < dataset.length; index++) {
			for(int indey = 0; indey < dataset[0][0].length; indey++) {
				if(dataset[index][0][indey] > inputMax[indey]) 
					inputMax[indey] = dataset[index][0][indey];
				
				if(dataset[index][0][indey] < inputMin[indey]) 
					inputMin[indey] = dataset[index][0][indey];
				
			}
			
			for(int indey = 0; indey < dataset[0][1].length; indey++) {
				if(dataset[index][1][indey] > outputMax[indey]) 
					outputMax[indey] = dataset[index][1][indey];
				
				if(dataset[index][1][indey] < outputMin[indey]) 
					outputMin[indey] = dataset[index][0][indey];
				
			}
		}
		
		//Lastly, perform normalization on each element of the dataset
		for(int index = 0; index < dataset.length; index++) {
			for(int indey = 0; indey < dataset[0][0].length; indey++) {
				dataset[index][0][indey] = normalize(dataset[index][0][indey], inputMax[indey], inputMin[indey],
						max, min);
			}
			
			for(int indey = 0; indey < dataset[0][1].length; indey++) {
				dataset[index][1][indey] = normalize(dataset[index][1][indey], outputMax[indey], outputMin[indey],
						max, min);
			}
		}
		
		return dataset;
	}
	
	/**
	 * Helper function to perform the actual normalization
	 * @param x Value to be normalized
	 * @param oldMax Maximum value from the dataset
	 * @param oldMin Minimum value from the dataset
 	 * @param newMax Desired maximum value
	 * @param newMin Desired minimum value
	 * @return Returns a normalized value within the range [newMax, oldMax]
	 */
	private static double normalize(double x, double oldMax, double oldMin, double newMax, double newMin) {
		double ret = (x - oldMin) / (oldMax - oldMin);
		ret *= (newMax - newMin);
		ret += newMin;
		
		return ret;
	}
	
	/**
	 * Function to perform the differencing transform onto a dataset.<br>
	 * Differencing is done by taking the discrete derivative of the data.
	 * In other words data[t] = data[t + 1] - data[t], or taking the "difference" from each datapoint.<br>
	 * By doing this, the dataset also loses the initial starting point for the data, but is able to make a dataset
	 * simpler to understand for a learning model.
	 * @param dataset dataset to be transformed
	 * @return transformed dataset
	 */
	public static double[][][] differenceTransform(double[][][] dataset){
		double[][][] ret = new double[dataset.length - 1][2][];
		
		for(int x = 1; x < dataset.length; x++) {
			ret[x - 1][0] = new double[dataset[x][0].length];
			ret[x - 1][1] = new double[dataset[x][1].length];
			
			for(int z = 0; z < dataset[x][0].length; z++) {
				ret[x - 1][0][z] = dataset[x][0][z] - dataset[x - 1][0][z];
			}
			
			for(int z = 0; z < dataset[x][1].length; z++) {
				ret[x - 1][1][z] = dataset[x][1][z] - dataset[x - 1][1][z];
			}
		}
		
		return ret;
	}
	
	/**
	 * Function to perform the reverse of the affects of differencing. This will be useful to accurately read data
	 * after the network has been trained.
	 * @param dataset differenced dataset to be returned to normal
	 * @param initialVals datapoint of the original initial values that were erased when differencing the dataset
	 * @return Reversed Difference transform of the dataset
	 */
	public static double[][][] reverseDifference(double[][][] dataset, double[][] initialVals) {
		double[][][] ret = new double[dataset.length + 1][2][];
		
		ret[0][0] = new double[initialVals[0].length];
		ret[0][1] = new double[initialVals[1].length];
		
		for(int arr = 0; arr < 2; arr++) {
			for(int i = 0; i < initialVals[arr].length; i++) {
				ret[0][arr][i] = initialVals[arr][i];
			}
		}
		
		for(int i = 1; i < ret.length; i++) {
			ret[i][0] = new double[initialVals[0].length];
			ret[i][1] = new double[initialVals[1].length];
			
			for(int z = 0; z < ret[i][0].length; z++) {
				ret[i][0][z] = ret[i - 1][0][z] + dataset[i - 1][0][z];
			}
			
			for(int z = 0; z < ret[i][1].length; z++) {
				ret[i][1][z] = ret[i - 1][1][z] + dataset[i - 1][1][z];
			}
		}
		
		return ret;
	}
	
	/**
	 * Function to create a dataset array with "Contextual Input"<br>
	 * Contextual Input involves having 2 distinct time steps as input with the goal to compute one new timestep
	 * as output.<br>
	 * Using this description an array with contextual input would have an input of {t[0], t[-1]}, and
	 * a target output of {t[1]}.<br>
	 * Using this function with with a typical dataset array consisting of {{input[t]}, {target[t + 1]}} 
	 * vectors, the first datapoint is erased to create an new array with the format 
	 * {{input[t], input[t - 1}, {target[t + 1]}}.<br>
	 * As an example, a dataset with { { {1}, {1} }, { {2}, {2} } will become { { { 2, 1}, {2} } }
	 * @param dataset Dataset to be transformed
	 * @param deepCopy Set to true to deep copy the dataset array
	 * @return transformed dataset array
	 */
	public static double[][][] contextualInput(double[][][] dataset, boolean deepCopy){
		if(deepCopy) dataset = deepCopy(dataset);
		
		double[][][] ret = new double[dataset.length - 1][2][];
		
		for(int i = 0; i < ret.length; i++) {
			ret[i][1] = dataset[i + 1][1];
			
			ret[i][0] = new double[dataset[i][0].length * 2];
			
			for(int z = 0; z < ret[i][0].length; z++) {
				int len = dataset[i][0].length;
				ret[i][0][z] = dataset[i + (z < len ? 1 : 0)][0][z % len];
			}
		}
		
		return ret;
	}
	
	/**
	 * Function to down-size a dataset.<br>
	 * Used by the network class to turn a dataset into a perfect multiple of the network's memory length.<br>
	 * If this function is used to attempt to up-size a dataset a null will be returned
	 * @param dataset Dataset to be resized
	 * @param newSize New size for the dataset
	 * @param deepCopy Set to true to deep copy the dataset array 
	 * @return The resized dataset
	 */
	public static double[][][] resize(double[][][] dataset, int newSize, boolean deepCopy){
		if(newSize > dataset.length) return null;
		
		if(deepCopy) dataset = deepCopy(dataset);
		double[][][] ret = new double[newSize][][];
		
		for(int i = 1; i <= newSize; i++) {
			ret[newSize - i] = dataset[dataset.length - i];
		}
		
		return ret;
	}
	
	/**
	 * Function to split a dataset into 2 separate datasets.<br>
	 * Used by the network class to create a difference between training and validation data.<br>
	 * If splitpoint is set to be larger than the dataset array already, a null will be returned
	 * @param dataset Dataset to be split
	 * @param splitPoint Spot to split the data; the datapoint at the splitpoint will be included in the second
	 * array as the first datapoint
	 * @param deepCopy Set to true to deep copy the dataset array
	 * @return Returns a 4D array of size 2 where ret[0] is the first dataset and ret[1] is the second
	 */
	public static double[][][][] splitDataset(double[][][] dataset, int splitPoint, boolean deepCopy){
		if(splitPoint > dataset.length) return null;
		
		if(deepCopy) dataset = deepCopy(dataset);
		
		int trainSize = splitPoint;
		int validateSize = dataset.length - splitPoint;
		
		double[][][] trainData = new double[trainSize][2][];
		double[][][] validateData = new double[validateSize][2][];
		
		for(int i = 0; i < trainSize; i++) {
			trainData[i] = dataset[i];
		}
		
		for(int i = 0; i < validateSize; i++) {
			validateData[i] = dataset[i + trainSize];
		}
		
		
		return new double[][][][] {trainData, validateData};
	}
	
}
