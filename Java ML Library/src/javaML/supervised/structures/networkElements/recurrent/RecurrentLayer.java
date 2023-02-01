package javaML.supervised.structures.networkElements.recurrent;

import javaML.supervised.Network;
import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;
import javaML.supervised.structures.networkElements.ffLayerTypes.HiddenLayer;

/**
 * The RecurrentLayer extension of the HiddenLayer class is representative of an advanced type of HiddenLayer.
 * In many ways, the RecurrentLayer is very similar to HiddenLayers in the fact that they cannot be directly
 * manipulated by the User, and there is no maximum on the number of RecurrentLayers that can be included.
 * <br>
 * However, RecurrentLayers have an additional input Matrix which represents a connection to this RecurrentLayer
 * from itself in a previous timestep.<br>
 * Effectively, this means all values from the layer in the previous timestep are passed forward through its own
 * weight matrix. 
 * 
 * @author Caleb Devon<br>
 * Created on 10/24/2022
 *
 */

public class RecurrentLayer extends HiddenLayer{
	
	/**
	 * Constructor for the RecurrentLayer extension of the HiddenLayer object
	 * @param layerSize Length of the vector
	 * @param memoryLength Number of time steps in which to save past data
	 * @param activationCode Identifier for which activation function to use (Use constants from the Network class
	 * to declare activation type)
	 * @param bias Boolean for whether or not to include a bias in the values vector
	 */
	public RecurrentLayer(int layerSize, int memoryLength, int activationCode, boolean bias) {
		super(layerSize, memoryLength, activationCode, bias);
	}

	@Override
	protected void calculateErrors(Vector errorVec, Matrix errorMat) {
		//dE/dY of a hidden layer by the chain rule will be equal to
		//i is a node of the hidden layer
		//j is a node of the next layer
		//dYj * Wij * activationDerivative
		Matrix conT = Matrix.transpose(errorMat);
		
		Vector currErrors = Matrix.multiply(conT, errorVec).getAsVector();
		currErrors = super.removeBias(currErrors);
		
		currErrors = Matrix.linearMultiply(currErrors, this.getRecentDerivatives()).getAsVector();
		
		super.addErrors(currErrors);
	}
	
	/**
	 * This function is responsible for correcting errors of the previous timestep.<br>
	 * Because errors are propagated backwards, as each new timestep occurs, there are background errors that
	 * need to be accounted for from the previous iteration.
	 * @param recMat Matrix used to determine the errors in the previous timestep
	 */
	protected void calcRecErrors(Matrix recMat) {
		Matrix conT = Matrix.transpose(recMat);
		
		Vector retroErrors = Matrix.multiply(conT, this.getRecentErrors()).getAsVector();
		retroErrors = super.removeBias(retroErrors);
		retroErrors = Matrix.linearMultiply(retroErrors, this.getDerivatives(memoryLength - 2)).getAsVector();
		
		super.adjustRecentErrors(retroErrors);
	}

	@Override
	public void runActivation() {
		this.activate();
	}
	
	@Override
	protected int getMemoryLength() {
		return this.memoryLength;
	}
	
	@Override
	protected Vector getValues(int index) {
		return super.getValues(index);
	}
	
	@Override
	public String toString() {
		return String.format("%d,%d,%d,%d\n", Network.RECURRENT, layerSize, activationCode, bias ? 1 : 0);
	}
	
}
