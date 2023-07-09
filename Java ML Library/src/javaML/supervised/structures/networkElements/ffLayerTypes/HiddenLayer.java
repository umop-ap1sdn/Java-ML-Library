package javaML.supervised.structures.networkElements.ffLayerTypes;

import javaML.supervised.Activation;
import javaML.supervised.LayerType;
import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;
import javaML.supervised.structures.networkElements.NeuronLayer;

/**
 * The HiddenLayer extension of the NeuronLayer class is representative of the layers between
 * the input and the output.
 * <br>
 * As its name suggests, HiddenLayers have no direct manipulation by the User after initialization
 * and instead serve only as intermediate values for the network.
 * <br><br>
 * Any given HiddenLayer will have 1 input Matrix and 1 output Matrix.<br>
 * User can determine if the HiddenLayer will use a bias and which activation function is used by
 * the layer.
 * <br><br>
 * 
 * Each Neural Network can have any number of HiddenLayers
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class HiddenLayer extends NeuronLayer {
	
	/**
	 * Constructor for the HiddenLayer extension of the NeuronLayer object
	 * @param layerSize Length of the vector
	 * @param memoryLength Number of time steps in which to save past data
	 * @param activationCode Identifier for which activation function to use (Use constants from the Network class
	 * to declare activation type)
	 * @param bias Boolean for whether or not to include a bias in the values vector
	 */
	public HiddenLayer(int layerSize, int memoryLength, Activation activationCode, boolean bias) {
		super(layerSize, memoryLength, activationCode, bias);
	}
	
	@Override
	public void calculateErrors(Vector errorVec, Matrix errorMat, int memIndex) {
		//dE/dY of a hidden layer by the chain rule will be equal to
		//i is a node of the hidden layer
		//j is a node of the next layer
		//dYj * Wij * activationDerivative
		
		Matrix conT = Matrix.transpose(errorMat);
		
		Vector currErrors = Matrix.multiply(conT, errorVec).getAsVector();
		currErrors = super.removeBias(currErrors);
		currErrors = Matrix.linearMultiply(currErrors, this.getDerivatives(memIndex)).getAsVector();
		
		super.addErrors(currErrors);
	}
	
	@Override
	public void runActivation() {
		this.activate();
	}

	@Override
	public String toString() {
		return String.format("%s,%d,%s,%d\n", LayerType.HIDDEN, layerSize, activationCode, bias ? 1 : 0);
	}
	
}
