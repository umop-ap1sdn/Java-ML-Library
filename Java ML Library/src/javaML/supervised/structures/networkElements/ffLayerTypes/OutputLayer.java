package javaML.supervised.structures.networkElements.ffLayerTypes;

import javaML.supervised.Network;
import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;
import javaML.supervised.structures.networkElements.NeuronLayer;

/**
 * The OutputLayer extension of the NeuronLayer class is representative of the exit point to the Neural Network.
 * <br><br>
 * The key properties of the OutputLayer include:<br>
 * - Its values are public to be retrieved by the User: this will be the final result of Neural Network operation.
 * <br>
 * - They lack a bias node; since this is the last layer of the network a potential bias node does not have a 
 * future layer to lead into, and therefore is unnecessary.<br>
 * - User can decide which activation function will be used by the OutputLayer; typical OutputLayer activations are
 * sigmoid and tanh.
 * <br><br>
 * 
 * Each Neural Network will have exactly 1 OutputLayer.
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class OutputLayer extends NeuronLayer {
	
	/**
	 * Constructor for the OutputLayer extension of the NeuronLayer class.<br>
	 * Constructor is nearly identical to its parent, but removes the bias parameter as the output layer has
	 * no need for a bias in its vector
	 * 
	 * @param layerSize Length of the vector
	 * @param memoryLength Number of time steps in which to save past data
	 * @param activationCode Identifier for which activation function to use (Use constants from the Network class
	 * to declare activation type)
	 */
	public OutputLayer(int layerSize, int memoryLength, int activationCode) {
		super(layerSize, memoryLength, activationCode, false);
	}
	
	/**
	 * Function to retrieve the final products of a neural network operation
	 * @return Array of outputs from the most recent time step
	 */
	public double[] getOutputs() {
		return this.getRecentValues().getVector();
	}
	
	//In the current version of the library, the Loss function will be predefined to be Mean Squared Error
	// 1/n * Summation[i = 0, n]((t-y)^2)
	@Override
	protected void calculateErrors(Vector errorVec, Matrix errorMat) {
		//errorVec will contain the targetValues for a given test case
		
		//Via the chain rule the error of an output layer neuron is given by the 
		//Partial derivative of E in respect to y multiplied by the derivative of the activation of y
		
		//dE/dy = 2/n(y - t)
				
		double scalar = 2.0 / this.getLayerSize();
		Vector baseErrors = Matrix.scale(errorVec, -1).getAsVector();
		baseErrors = Matrix.add(this.getRecentValues(), baseErrors).getAsVector();
		baseErrors = Matrix.scale(baseErrors, scalar).getAsVector();
		
		baseErrors = Matrix.linearMultiply(baseErrors, this.getRecentDerivatives()).getAsVector();
		
		super.addErrors(baseErrors);
		super.putErrors();
	}
	
	@Override
	public void runActivation() {
		this.activate();
	}

	@Override
	public String toString() {
		return String.format("%d,%d,%d,%d\n", Network.OUTPUT, layerSize, activationCode, bias ? 1 : 0);
	}
	
}
