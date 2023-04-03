package javaML.supervised;

import javaML.DataTransformations;
import javaML.supervised.structures.networkElements.*;
import javaML.supervised.structures.networkElements.ffLayerTypes.*;
/**
 * 
 * The network class is the primary controller of the supervised learning package<br>
 * This class is responsible for running all of the routines including:
 * forward pass, loss calculation, error propagation, and backpropagation<br><br>
 * 
 * This class can only be constructed by the NetworkBuilder class and can be
 * attained by the user by following the steps of the NetworkBuilder class up to the build(int) function
 * 
 * @author Caleb Devon<br>
 * Created 10/14/2022
 * 
 */

public class Network {
	
	private double learning_rate = 0.02;
	
	private InputLayer input;
	private OutputUnit output;
	private Unit[] hiddenLayers;
	private int numHidden;
	
	private int batchSize;
	private int memoryLength;
	
	private double[][][] dataset;
	private double[][][] validationData;
	
	
	private int dataIndex = 0, dataSize = 0;
	
	private double totalLoss = 0, averageLoss = 0;
	
	/**
	 * Constructor for use only by the NetworkBuilder class
	 * @param input Input Layer
	 * @param output Output Layer
	 * @param hiddenLayers Array for Hidden Layers
	 * @param cLayers Array for Connection Layers
	 * @param batchSize value for number of training steps between pauses
	 * @param memoryLength value for how deep the memory of neuron layers go
	 */
	protected Network(InputLayer input, OutputUnit output, Unit[] hiddenLayers, int batchSize, int memoryLength) {
		this.input = input;
		this.output = output;
		this.hiddenLayers = hiddenLayers;
		this.batchSize = batchSize;
		this.memoryLength = memoryLength;
		
		this.numHidden = hiddenLayers.length;
		
		this.dataset = null;
		this.validationData = null;
		
	}
	
	/**
	 * Sets a new value to the learning rate<br>
	 * Defaults to 0.02 upon construction
	 * @param lr new learning rate
	 */
	public void setLearningRate(double lr) {
		this.learning_rate = lr;
	}
	
	
	//TODO add dataset uploading options for trend removing using differencing
	//TODO add separation of dataset into training set and test set
	
	/**
	 * Use this function to upload a user made dataset<br>
	 * In order for training to work properly data should be formatted such that
	 * { { {Input Data 2}, {Target Output Data 1} }, { {Input Data 2}, {Target Output Data 2} }, ... }
	 * @param dataset dataset to be loaded
	 * @param normalizeCode Identifier which states how the dataset will be normalized (if at all).<br>
	 * Use the Normalize enum to declare which normalization function to use.<br>
	 * Use Normalize.NONE_NORMALIZE, Normalize.SIGMOID_NORMALIZE or Normalize.TANH_NORMALIZE to specify normalization
	 * function
	 */
	public void uploadDataset(double[][][] dataset, Normalize normalizeCode) {
		int resize = dataset.length - (dataset.length % batchSize);
		
		double[][][] data = DataTransformations.resize(dataset, resize, false);
		
		
		if(normalizeCode == Normalize.SIGMOID_NORMALIZE) 
			data = DataTransformations.normalize(data, 1, 0, true);
		if(normalizeCode == Normalize.TANH_NORMALIZE) 
			data = DataTransformations.normalize(data, 1, -1, true);
		
		double[][][][] placeHolder = DataTransformations.splitDataset(data, data.length - (batchSize * 2), false);
		
		this.dataset = placeHolder[0];
		this.validationData = placeHolder[1];
		
		
		this.dataIndex = 0;
		this.dataSize = this.dataset.length;
	}
	
	
	/**
	 * Primary algorithm to be called by the user to train the network<br>
	 * Network will run through the dataset uploaded to run the forwardPass(), propagateErrors(),
	 * and backpropagate() functions
	 * @param backProp set to true if backpropagation is desired
	 * @param dependency if set to true, backpropagation will be restricted if memory has not been filled
	 * with relevant data<br>
	 * Will be used with RNN's that should only be backpropagated if the memory is full
	 * @param batchSize the amount of training steps to run in a particular train routine
	 * @return Returns true if an overflow occured when reading through the data<br>
	 * Will be useful for RNN's
	 */
	public boolean train(boolean backProp, boolean dependency, int batchSize) {
		if(dataset == null) return false;
		
		totalLoss = 0;
		averageLoss = 0;
		
		boolean overflow = false;
		
		double[] output;
		
		//Run through the dataset
		for(int index = 0; index < batchSize; index++) {
			
			//Test data, calculate errors, add errors to loss value
			output = test(dataset[dataIndex][0]);
			calculateLoss(dataset[dataIndex][1], output);
			
			//ensure dataIndex never reaches out of bounds for the dataset array
			dataIndex = (dataIndex + 1) % dataSize;
			
			//When data is reset an overflow has occurred
			if(dataIndex == 0 && index < batchSize - 1) overflow = true;
		}
		
		propagateError();
		
		//Backpropagate if desired
		if(backProp && (!dependency || dataIndex >= memoryLength || dataIndex == 0)) {
			backpropagate();
		}
		
		averageLoss = totalLoss / batchSize;
		
		return overflow;
	}
	
	public double trainEpoch(boolean backProp, boolean dependency) {
		for(int i = 0; i * batchSize < dataSize; i++) {
			train(backProp, dependency);
		}
		
		double loss = validate();
		this.reset(true);
		
		return loss;
	}
	
	/**
	 * Basic Train Function that defaults to the specified batch size when training
	 * @param backProp boolean for whether to backpropagation or not
	 * @param dependency boolean to restrict backpropagation if the full memory of the network is not filled
	 * by the end of the training session
	 * @return Returns true if an overflow occurred when reading through the data<br>
	 * Will be useful for RNN's 
	 */
	public boolean train(boolean backProp, boolean dependency) {
		return train(backProp, dependency, batchSize);
	}
	
	public double validate() {
		totalLoss = 0;
		averageLoss = 0;
		double[] output;
		
		for(double[][] data: validationData) {
			output = test(data[0]);
			calculateLoss(data[1], output);
		}
		
		averageLoss = totalLoss / batchSize;
		
		return averageLoss;
	}
	
	/**
	 * Function to test a particular set of inputs
	 * @param inputs Input vector of size equal to input layer size
	 * @return returns the array of outputs produced by the output layer through forward propagation
	 */
	public double[] test(double... inputs) {
		forwardPass(inputs);
		return output.getOutputs();
	}
	
	/**
	 * Function to reset the values of each Neuron Layer.<br>
	 * This will be useful for RNN's whose outputs are influenced by what is already stored in Neuron
	 * Layer memory
	 * 
	 * @param resetDataIndex when set to true dataIndex will be set back to 0
	 */
	public void reset(boolean resetDataIndex) {
		input.reset();
		output.reset();
		for(int index = 0; index < hiddenLayers.length; index++) {
			hiddenLayers[index].reset();
		}
		
		if(resetDataIndex) dataIndex = 0;
	}
	
	/**
	 * Function to be called only natively by the Network class<br>
	 * Runs the forward propagation algorithm for all layers
	 * @param inputs array of inputs to be given to the input layer
	 */
	private void forwardPass(double[] inputs) {
		input.setInputs(inputs);
		input.runActivation();
		for(int index = 0; index < hiddenLayers.length; index++) {
			hiddenLayers[index].forwardPass();
			hiddenLayers[index].runActivation();
		}
		output.forwardPass();
		output.runActivation();
	}
	
	/**
	 * Function to completely propagate errors at all timesteps.<br>
	 * Will be called after the entire batch has been passed through as input
	 */
	private void propagateError() {
		int dataPoint = dataIndex - 1; //Most Recent trained point
		if(dataPoint < 0) dataPoint += dataSize;
		
		int memIndexSrt = memoryLength - batchSize;
		
		for(int step = batchSize - 1; step >= 0; step--) {
			propagateErrorStep(memIndexSrt + step, dataset[dataPoint][1]);
			
			//System.out.println(memIndexSrt + step + ", " + dataPoint);
			
			dataPoint--;
			if(dataPoint < 0) dataPoint += dataSize;
		}
	}
	
	/**
	 * Function to be called only natively by the Network class<br>
	 * Runs the calculateError function for each Neuron Layer
	 * @param memIndex Location (timestep) in memory from which to calculate error from.
	 * @param target array for the target values for a particular set of inputs
	 */
	private void propagateErrorStep(int memIndex, double[] target) {
		output.setTargets(target);
		output.calcErrors(null, memIndex);
		
		if(numHidden >= 1) {
			hiddenLayers[numHidden - 1].calcErrors(output, memIndex);
			for(int index = numHidden - 2; index >= 0; index--) {
				hiddenLayers[index].calcErrors(hiddenLayers[index + 1], memIndex);
			}
		}
		
		input.calculateErrors(target, null, 0);
	}
	
	/**
	 * Function to be called only natively by the Network class<br>
	 * Runs the adjustWeights function for each layer of connections
	 */
	private void backpropagate() {
		output.backpropagation(learning_rate);
		output.purgeErrors(batchSize);			// Purge errors after backpropagation
		for(Unit u: hiddenLayers) {
			u.backpropagation(learning_rate);
			u.purgeErrors(batchSize);			// Purge after backprop
		}
		
		input.purgeErrors(batchSize);
	}
	
	/**
	 * Function to be called only natively by the Network class<br>
	 * Runs the MSE (Mean Squared Error) algorithm to calculate overall loss of a batch
	 * @param target array of target values
	 * @param output array of output values
	 */
	private void calculateLoss(double[] target, double[] output) {
		//1/n (t - y)^2
		//n = number of output neurons
		//t is the target
		//y is the output
		double sum = 0;
		for(int index = 0; index < target.length; index++) {
			sum += Math.pow(target[index] - output[index], 2);
		}
		
		sum /= target.length;
		
		totalLoss += sum;
	}
	
	/**
	 * Function to get the overall loss of a batch
	 * @return calculated loss
	 */
	public double getTotalLoss() {
		return this.totalLoss;
	}
	
	/**
	 * Function to get the average loss per element in a batch
	 * @return totalLoss / batchSize
	 */
	public double getAverageLoss() {
		return this.averageLoss;
	}
	
	@Override
	public String toString() {
		String ret = String.format("%s,%s\n", memoryLength, batchSize);
		ret += input + "-\n";
		
		for(Unit u: hiddenLayers) ret += u;
		ret += output + "---";
		
		return ret;
	}
}