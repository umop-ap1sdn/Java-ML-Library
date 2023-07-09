package javaML.supervised.structures.networkElements.ffLayerTypes;

import javaML.supervised.structures.networkElements.ConnectionLayer;
import javaML.supervised.structures.networkElements.Unit;

/**
 * The OutputUnit extension of the Unit class is designed to oversee all aspects of the OutputLayer.<br>
 * As such, it is responsible for managing all parts of the OutputLayer, such as the forward pass, and is the
 * starting point for backpropagation.<br>
 * In addition, the OutputUnit class is able to return the data from the OutputLayer to send data back to the
 * User after an iteration of Neural Network operation has completed.
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class OutputUnit extends Unit {
	
	private final OutputLayer layer;
	private double[] targets;
	
	/**
	 * Constructor uses the basic structure for Unit Construction consisting of just one OutputVector and one 
	 * input Matrix
	 * @param layer OutputLayer
	 * @param conIn ConnectionLayer input
	 */
	public OutputUnit(OutputLayer layer, ConnectionLayer conIn){
		super(layer, conIn);
		this.layer = layer;
	}
	
	@Override
	public void forwardPass() {
		cLayers[0].forwardPass();
	}

	@Override
	public void calcErrors(Unit next, int memIndex) {
		//Output layer will always be the final layer, therefore next = null
		layer.calculateErrors(targets, null, memIndex);
		
	}
	
	/**
	 * Function to set the targets array.<br>
	 * Will be called by the Network class during every iteration to set the targets for every given input vector
	 * @param targets Array of target values to be set
	 */
	public void setTargets(double[] targets) {
		this.targets = targets;
	}
	
	/**
	 * Function to return the calculated outputs at the end of Neural Network operations
	 * @return Array vector of the calculated output values
	 */
	public double[] getOutputs() {
		return layer.getOutputs();
	}
}
