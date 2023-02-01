package javaML.supervised.structures.networkElements;

import javaML.supervised.structures.*;
import javaML.supervised.structures.networkElements.ffLayerTypes.*;
import javaML.supervised.structures.networkElements.recurrent.RecurrentLayer;

/**
 * The ConnectionLayer class represents the layers of weights that exist between each NeuronLayer
 * They consist of an source, destination, and a Matrix, the source being where data originates, the
 * destination being where data is delivered and the Matrix being responsible for the transformation of the
 * input data.
 * <br><br>
 * The ConnectionLayer is the primary "learning" component of a Neural Network. When backpropagation is performed
 * the ConnectionLayers will use the associated error with it's respective destination NeuronLayer to adjust
 * the weights within its Matrix. These values are constantly shifted throughout training to create a solution
 * for all training inputs.
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class ConnectionLayer {
	
	protected int destSize, sourceSize;
	
	protected final NeuronLayer source;
	protected final NeuronLayer destination;
	protected Matrix layer;
	
	//Nearly Identical constructors that each control how ConnectionLayers are allowed to be defined
	
	/**
	 * Constructor for a ConnectionLayer leading from Input to Hidden Layer
	 * @param source Input Layer
	 * @param destination Hidden Layer
	 */
	public ConnectionLayer(InputLayer source, HiddenLayer destination) {
		this.source = source;
		this.destination = destination;
		initialize();
	}
	
	/**
	 * Constructor for a ConnectionLayer leading from Hidden to Hidden Layers
	 * @param source Hidden Layer
	 * @param destination Hidden Layer
	 */
	public ConnectionLayer(HiddenLayer source, HiddenLayer destination) {
		this.source = source;
		this.destination = destination;
		initialize();
	}
	
	/**
	 * Constructor for a ConnectionLayer leading from Hidden to Output Layer
	 * @param source Hidden Layer
	 * @param destination Output Layer
	 */
	public ConnectionLayer(HiddenLayer source, OutputLayer destination) {
		this.source = source;
		this.destination = destination;
		initialize();
	}
	
	/**
	 * Constructor for a ConnectionLayer leading from and to a Recurrent Layer
	 * @param layer Recurrent Layer
	 */
	protected ConnectionLayer(RecurrentLayer layer) {
		this.source = layer;
		this.destination = layer;
		initialize();
	}
	
	/**
	 * Constructor for a Connection Layer leading from Input to Output Layer<br>
	 * Recommended not to be used, as this implies the nonexistence of hidden layers of any kind
	 * @param source Input Layer
	 * @param destination Output Layer
	 */
	public ConnectionLayer(InputLayer source, OutputLayer destination) {
		this.source = source;
		this.destination = destination;
		initialize();
	}
	
	/**
	 * Basic Function for initializing the connection layers
	 */
	private void initialize() {
		
		// NOTE: getLayerSize() returns the the normal size of a NeuronLayer
		// getTrueSize() returns the size of the layer plus a bias that may or may not exist
		// bias is not effected by the previous layer but does influence the next layer which is why the destSize
		// does not include the bias while the source does
		
		destSize = destination.getLayerSize();
		sourceSize = source.getTrueSize();
		
		// Initialize all matrix values to random
		layer = new Matrix(destSize, sourceSize, Matrix.FILL_RANDOM);
	}
	
	/**
	 * Function to perform the forward pass between the source layer and the destination layer
	 */
	public void forwardPass() {
		
		// Forward pass is achieved by performing a matrix multiplication between the the ConnectionLayer matrix
		// and the source NeuronLayer vector which produces a vector of size needed for the destination
		// NeuronLayer
		Vector result = Matrix.multiply(layer, source.getRecentValues()).getAsVector();
		destination.pushValues(result);
	}
	
	/**
	 * This function performs the last step of the backpropagation process.<br>
	 * It takes the result of error calculation and recent activation values to adjust the weight matrix
	 * of the ConnectionLayer
	 * @param lr (learning rate) is a scalar (typically <= 0.1) that determines how much to adjust
	 * weights. Higher learning rates typically lead to faster learning but lower precision and vice versa
	 */
	public void adjustWeights(final double lr) {
		Matrix gradients = new Matrix(layer.getRows(), layer.getColumns(), Matrix.FILL_ZERO);
		
		// Triple nested for loop :( O(n^3)
		for(int row = 0; row < destSize; row++) {
			for(int col = 0; col < sourceSize; col++) {
				double gradient = 0;
				
				// Last loop goes through NeuronLayer memory to backpropagate through all timesteps still in memory
				for(int count = 0; count < source.getMemoryLength(); count++) {
					
					// Gradient is equal to the sum of source values multiplied by the destination errors
					gradient += source.getValues(count).getValue(col) * destination.getErrors(count).getValue(row);
				}
				
				// Scale by lr
				gradient *= -lr;
				gradients.setValue(gradient, row, col);
			}
		}
		
		layer = Matrix.add(layer, gradients);
	}
	
	/**
	 * Function to get the source NeuronLayer
	 * @return Source
	 */
	public NeuronLayer getSource() {
		return this.source;
	}
	
	/**
	 * Function to get the destination NeuronLayer
	 * @return Destination
	 */
	public NeuronLayer getDestination() {
		return this.destination;
	}
	
	/**
	 * Function to get the ConnectionLayer Matrix
	 * @return Layer Matrix
	 */
	public Matrix getMatrix() {
		return this.layer;
	}
	
	/**
	 * Function to set the layer matrix
	 * @param arr 2D array form of new Matrix
	 */
	public void setMatrix(double[][] arr) {
		this.layer = new Matrix(arr);
	}
	
	@Override
	public String toString() {
		return layer.toString() + "\n";
	}
}
