package javaML.supervised.structures.networkElements.recurrent.gru;

import java.util.LinkedList;

import javaML.supervised.Activation;
import javaML.supervised.structures.Vector;
import javaML.supervised.structures.networkElements.ffLayerTypes.HiddenLayer;

/**
 * GRU_InternalLayer is an abstract class meant for housing the Neuron Layers within the GRU Cell.<br>
 * These layers consist of the Update Layer (or update gate), Reset Layer (reset gate), and the Intermediate Layer 
 * (typically denoted as ~h_t).<br>
 * Unlike most Neuron Layers, they often have additional calculations performed after the normal forward propagation
 * process.
 * @author Caleb Devon<br>
 * Created on 5/2/2023
 *
 */

public abstract class GRU_InternalLayer extends HiddenLayer {
	
	//intermediateVals denotes the value calculated from a process that takes place after forward propagation
	protected LinkedList<Vector> intermediateVals;
	//hiddenState refers to values sent by other layers to be used in intermediate calculations
	protected Vector hiddenState;
	
	/**
	 * Primary Constructor for the GRU_InternalLayer object<br>
	 * By default GRU_Internal layers will use the Sigmoid Activation function, and will have a bias.
	 * @param layerSize Number of Neurons in the layer
	 * @param memoryLength Number of epochs to be iterated through during training
	 */
	protected GRU_InternalLayer(int layerSize, int memoryLength) {
		super(layerSize, memoryLength, Activation.SIGMOID, true);
	}
	
	/**
	 * Secondary Constructor for the GRU_InternalLayer object<br>
	 * This constructor will allow for a pre-specified Activation function, however bias is still always set to true.
	 * @param layerSize Number of Neurons in the layer
	 * @param memoryLength Number of epochs to be iterated through during training
	 * @param activationCode Activation function to be used by the layer
	 */
	protected GRU_InternalLayer(int layerSize, int memoryLength, Activation activationCode) {
		super(layerSize, memoryLength, activationCode, true);
	}
	
	/**
	 * Specialized initialize function for the extra vectors used by this type of layer
	 */
	protected void initialize() {
		intermediateVals = new LinkedList<>();
		hiddenState = new Vector(layerSize, Vector.FILL_ZERO);
		
		for(int i = 0; i < memoryLength; i++) {
			intermediateVals.add(new Vector(layerSize, Vector.FILL_ZERO));
		}
	}
	
	@Override
	protected Vector getValues(int index) {
		return super.getValues(index);
	}
	
	/**
	 * Function to be called by other elements of the GRU cell, allows for communicating vectors which will be used
	 * in intermediate calculations.
	 * @param hiddenState Vector of values to be used in intermediate calculation.
	 */
	protected void setHiddenState(Vector hiddenState) {
		this.hiddenState = new Vector(hiddenState.getVector());
	}
	
	@Override
	public void reset() {
		super.reset();
		
		//Include new initialize function in the reset process
		this.initialize();
	}
	
	/**
	 * Function to get the results of intermediate calculation
	 * @return List of intermediate vectors at each timestep in memory
	 */
	protected LinkedList<Vector> getIntermediateVals() {
		return intermediateVals;
	}
	
	/**
	 * Function to get the most recent result of intermediate calculation
	 * @return Most recently calculated result vector
	 */
	protected Vector getRecentIntermediateVals() {
		return padBias(intermediateVals.getLast());
	}
}
