package javaML.supervised.structures.networkElements.recurrent.gru;

import javaML.supervised.Activation;
import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;

/**
 * The Intermediate Layer of the GRU Cell represents the last NeuronLayer of the
 * GRU_InternalLayers and is responsible for combining the results of the reset,
 * update and input layers into one send-able packet to the output layer.
 * 
 * 
 * @author Caleb Devon<br>
 * Created on 5/2/2023
 */

public class IntermediateLayer extends GRU_InternalLayer {
	
	private Vector update;
	
	/**
	 * Constructor for the Intermediate Layer<br>
	 * Activation Function and Bias are pre-determined to be TANH and true
	 * respectively.
	 * @param layerSize Number of Neurons in the layer
	 * @param memoryLength Number of epochs for each training step
	 */
	protected IntermediateLayer(int layerSize, int memoryLength) {
		super(layerSize, memoryLength, Activation.TANH);
		initialize();
	}
	
	/**
	 * Because further calculations are done at each layer, additional
	 * error calculations also need to be done using data not native to this
	 * layer.<br>
	 * For the intermediate layer, its error vector is the output vector of the
	 * update gate.
	 * @param update auxiliary vector containing values for error calculation
	 */
	protected void setErrorVector(Vector update) {
		this.update = new Vector(update.getVector());
	}

	@Override
	public void calculateErrors(Vector errorVec, Matrix errorMat, int memIndex) {
		update = super.removeBias(update);
		
		Vector error = Vector.linearMultiply(errorVec, update).getAsVector();
		error = Vector.linearMultiply(error, this.getDerivatives(memIndex)).getAsVector();
		
		super.addErrors(error);
	}

	@Override
	public void runActivation() {
		super.activate();
		
		Vector intermediate = Vector.linearMultiply(removeBias(hiddenState), activations.getLast()).getAsVector();
		intermediateVals.addLast(intermediate);
		intermediateVals.pollFirst();
		
	}

}
