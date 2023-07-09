package javaML.supervised.structures.networkElements.recurrent.gru;

import javaML.supervised.Activation;
import javaML.supervised.LayerType;
import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;

/**
 * The Reset Layer is one of the initial layers to the GRU Cell.<br>
 * It is responsible mostly for erasing unimportant details from 
 * memory, and one of the reasons it is said to have solved the 
 * vanishing gradient problems of vanilla recurrent neural networks.
 * 
 * 
 * @author Caleb Devon<br>
 * Created on 5/2/2023
 */

public class ResetLayer extends GRU_InternalLayer {
	
	private Vector previous;
	
	/**
	 * Constructor for the Reset Layer<br>
	 * Activation Function and Bias are pre-determined to be Sigmoid and true
	 * respectively.
	 * @param layerSize Number of Neurons in the layer
	 * @param memoryLength Number of epochs for each training step
	 */
	protected ResetLayer(int layerSize, int memoryLength) {
		super(layerSize, memoryLength);
		initialize();
	}
	
	/**
	 * Because further calculations are done at each layer, additional
	 * error calculations also need to be done using data not native to this
	 * layer.<br>
	 * For the reset layer, its error vector is the output vector of the
	 * output layer for the previous timestep.
	 * @param previous auxiliary vector containing values for error calculation
	 */
	protected void setErrorVector(Vector previous) {
		this.previous = new Vector(previous.getVector());
	}
	
	@Override
	public void calculateErrors(Vector errorVec, Matrix errorMat, int memIndex) {
		
		previous = super.removeBias(previous);
		
		Matrix conT = Matrix.transpose(errorMat);
		
		Vector currErrors = Matrix.multiply(conT, errorVec).getAsVector();
		currErrors = super.removeBias(currErrors);
		currErrors = Vector.linearMultiply(currErrors, previous).getAsVector();
		currErrors = Vector.linearMultiply(currErrors, this.getDerivatives(memIndex)).getAsVector();
		
		super.addErrors(currErrors);
		
	}

	@Override
	public void runActivation() {
		// TODO Auto-generated method stub
		super.activate();
		
		Vector intermediate = Vector.linearMultiply(removeBias(hiddenState), activations.getLast()).getAsVector();
		intermediateVals.addLast(intermediate);
		intermediateVals.pollFirst();
		
	}
	
	@Override
	public String toString() {
		return String.format("%s,%d,%s,%d\n", LayerType.GRU, layerSize, Activation.INVALID, 0);
	}

}
