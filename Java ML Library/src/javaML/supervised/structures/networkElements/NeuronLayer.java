package javaML.supervised.structures.networkElements;

import java.util.LinkedList;
import javaML.supervised.Network;
import javaML.supervised.structures.*;

/**
 * The NeuronLayer class is designed to model the Neurons in a Neural Network. This is an abstract class that
 * describes the general structure for a NN Layer.
 * <br><br>
 * NeuronLayers are modeled as vectors, as they are a collection of values. 
 * Additionally they are also responsible for storing their historical error values. There are several types of
 * NLayers including Input, Hidden, Output and more. Each of them have the same basic functionality but all unique
 * layer types have their own exclusive features as well.
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */
public abstract class NeuronLayer {
	
	/**
	 * Memory Length is a parameter that represents how many time states back in which data will be saved
	 */
	protected final int memoryLength;
	protected final int activationCode;
	protected final int layerSize;
	protected final int trueSize;
	
	protected final boolean bias;
	
	private LinkedList<Vector> activations;
	private LinkedList<Vector> derivatives;
	private LinkedList<Vector> errors;
	
	Vector unactivated;
	Vector errorVec;
	
	ActivationFunctions function;
	
	/**
	 * Primary constructor for the NeuronLayer class type
	 * Because this is an abstract class, this constructor will be called only by extensions of this class
	 * @param layerSize Length of the vector
	 * @param memoryLength Number of time steps in which to save past data
	 * @param activationCode Identifier for which activation function to use (Use constants from the Network class
	 * to declare activation type)
	 * @param bias Boolean for whether or not to include a bias in the values vector
	 */
	protected NeuronLayer(int layerSize, int memoryLength, int activationCode, boolean bias) {
		this.layerSize = layerSize;
		this.memoryLength = memoryLength;
		this.activationCode = activationCode;
		this.bias = bias;
		
		this.trueSize = layerSize + (bias ? 1 : 0);
		
		initializeLists();
		initializeActivation();
	}
	
	/**
	 * Function used to calculate the errors of a particular layer.<br>
	 * Note that the input layer will not have an error.
	 * @param errorVec Multiple purpose vector used for the determination of error values<br>
	 * In the output layer this vector will be the target values, while most other layers will use the errors from
	 * the next layer.
	 * @param errorMat Matrix used to help determine the error of a given layer
	 */
	protected abstract void calculateErrors(Vector errorVec, Matrix errorMat);
	
	/**
	 * Function to run the activation of a particular layer<br>
	 * In most layers this will be a simple function while more complicated layers might have more complex
	 * processes
	 */
	public abstract void runActivation();
	
	@Override
	public abstract String toString();
	
	/**
	 * Function to initialize the lists for activations, derivatives, and errors.<br>
	 * Called during construction and during the reset() function.
	 */
	private void initializeLists() {
		
		activations = new LinkedList<>();
		derivatives = new LinkedList<>();
		errors = new LinkedList<>();
		
		for(int index = 0; index < memoryLength; index++) {
			activations.add(new Vector(layerSize, Matrix.FILL_ZERO));
			derivatives.add(new Vector(layerSize, Matrix.FILL_ZERO));
			errors.add(new Vector(layerSize, Matrix.FILL_ZERO));
		}
	}
	
	/**
	 * Function called only during construction.<br>
	 * Helper function that initializes the functionality of the activation function
	 */
	private void initializeActivation() {
		switch(activationCode) {
		case Network.LINEAR:
			function = new Linear();
			break;
		case Network.RELU:
			function = new ReLU();
			break;
		case Network.SIGMOID:
			function = new Sigmoid();
			break;
		case Network.TANH:
			function = new Tanh();
			break;
		default:
			function = new Linear();
			break;
		}
		
		unactivated = new Vector(layerSize, Matrix.FILL_ZERO);
		errorVec = new Vector(layerSize, Matrix.FILL_ZERO);
	}
	
	/**
	 * Public access way to reset the memory values in each list
	 */
	public void reset() {
		this.initializeLists();
	}
	
	/**
	 * Function called during forward pass to pass information from layers connecting into this one.<br>
	 * This function does not activate values in as some cases layers have multiple input vectors that all need
	 * to be accounted for before activation occurs
	 * @param values Vector that represents the result of matrix multiplication from the source layer
	 */
	public void pushValues(Vector values) {
		unactivated = Matrix.add(unactivated, values).getAsVector();
	}
	
	/**
	 * Function to activate the currently loaded values.<br>
	 * This is the last step of the forward pass and should only be used when all inputs into a layer have been
	 * passed.
	 */
	public void activate() {
		double[] arr = unactivated.getVector();
		
		derivatives.addLast(new Vector(ActivationFunctions.derivative(function, arr)));
		activations.addLast(new Vector(ActivationFunctions.activate(function, arr)));
		
		activations.pollFirst();
		derivatives.pollFirst();
		
		unactivated = new Vector(layerSize, Matrix.FILL_ZERO);
		
	}
	
	/**
	 * Shortcut function to calculate errors based on arrays instead of vectors and matrices.
	 * @param errors 1D array (Vector) for the errorVec
	 * @param matrix 2D array (Matrix) for the errorMat
	 */
	public void calculateErrors(double[] errors, double[][] matrix) {
		if(matrix == null) this.calculateErrors(new Vector(errors), null);
		else this.calculateErrors(new Vector(errors), new Matrix(matrix));
	}
	
	/**
	 * Function to get a value at a particular time step
	 * @param index Time step for which to receive values from
	 * @return Vector of the Neuron Values at the specified time step
	 */
	protected Vector getValues(int index) {
		if(bias) return padBias(activations.get(index));
		else return activations.get(index);
	}
	
	/**
	 * Shortcut function to get the values at the most recent time step
	 * @return Vector of the neuron values at the most recent time step
	 */
	protected Vector getRecentValues() {
		if(bias) return padBias(activations.getLast());
		else return activations.getLast();
	}
	
	/**
	 * Function to be called only by the native class.<br>
	 * Takes a vector that represents the activations of a NeuronLayer and adds a 1 valued element to the end of
	 * the vector. This is done to represent the bias that may or may not exist on a particular layer.
	 * @param vec
	 * @return
	 */
	private Vector padBias(Vector vec) {
		double[] vector = new double[trueSize];
		double[] vecArr = vec.getVector();
		
		for(int index = 0; index < vecArr.length; index++) {
			vector[index] = vecArr[index];
		}
		
		//if layer does not have a bias return identical vector
		if(!bias) return new Vector(vector);
		
		vector[layerSize] = 1;
		
		return new Vector(vector);
	}
	
	/**
	 * Function to remove from a list of activations.<br>
	 * Returns the input if the the NeuronLayer has no bias by default.<br>
	 * This function will be used during backpropagation to remove biases that do not properly fit into transposed
	 * matrices
	 * @param vec Vector of inputs to be operated on
	 * @return Returns the vector with the bias removed, returns an identical vector if there was no bias.
	 */
	protected Vector removeBias(Vector vec) {
		//Skip running the function if there is no bias
		if(!bias) return vec;
		
		Vector newVec = new Vector(layerSize, Vector.FILL_ZERO);
		for(int index = 0; index < layerSize; index++) {
			newVec.setValue(vec.getValue(index), index);
		}
		
		return newVec;
	}
	
	/**
	 * Function to add errors to the current errorVec.<br>
	 * Similar to the pushValues() function this does not finalize the errors to account for layers that may have
	 * multiple inputs
	 * @param errorVec Vector to add to the current vector of errors
	 */
	public void addErrors(Vector errorVec) {
		this.errorVec = Vector.add(this.errorVec, errorVec);
	}
	
	/**
	 * Function to add finalize and add the errorVec to the list of memorized error values
	 */
	public void putErrors() {
		errors.addLast(errorVec);
		errors.pollFirst();
		
		errorVec = new Vector(layerSize, Matrix.FILL_ZERO);
	}
	
	/**
	 * Function to get the error vector from a given time step
	 * @param index Time step for which to get errors from
	 * @return Vector of the errors at the specified time step
	 */
	public Vector getErrors(int index) {
		return errors.get(index);
	}
	
	/**
	 * Shortcut function to get the error vector at the most recent time step
	 * @return Vector of the most recent error values
	 */
	public Vector getRecentErrors() {
		return errors.getLast();
	}
	
	/**
	 * Function to adjust recently calculated error values.<br>
	 * Will be used for recurrent types of neural networks
	 * @param errorVec Vector of errors to adjust recent values
	 */
	protected void adjustRecentErrors(Vector errorVec) {
		int recentIndex = memoryLength - 2;
		errors.set(recentIndex, Vector.add(errorVec, errors.getLast()));
	}
	
	/**
	 * Function to get the derivative vector from a given time step
	 * @param index Time step for which to get the derivative from
	 * @return Vector of the derivative at the specified time step
	 */
	protected Vector getDerivatives(int index) {
		return derivatives.get(index);
	}
	
	/**
	 * Shortcut function to get the derivative vector at the most recent time step
	 * @return Vector of the most recent derivative values
	 */
	protected Vector getRecentDerivatives() {
		return derivatives.getLast();
	}
	
	/**
	 * Function to get the size of the standardized (no bias) layer
	 * @return Size of the layer excluding bias
	 */
	protected int getLayerSize() {
		return this.layerSize;
	}
	
	/**
	 * Function to get the true size of the layer (bias)
	 * @return Size of the layer including bias
	 */
	protected int getTrueSize() {
		return this.trueSize;
	}
	
	/**
	 * Function to get the length for which memory is saved in the Layer
	 * @return Memory Length
	 */
	protected int getMemoryLength() {
		return this.memoryLength;
	}
}