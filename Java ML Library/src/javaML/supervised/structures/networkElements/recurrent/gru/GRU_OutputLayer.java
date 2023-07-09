package javaML.supervised.structures.networkElements.recurrent.gru;

import javaML.supervised.Activation;
import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;
import javaML.supervised.structures.networkElements.recurrent.RecurrentLayer;

/**
 * The GRU_OutputLayer is the only GRU specific layer not to be categorized as
 * an internal layer. This is because it connects directly to the next Unit
 * within the network, and connects back to the start of the GRU in the
 * next timestep.<br>
 * The GRU_OutputLayer is very unique in that it does not have any explicit
 * input ConnectionLayers; instead, its values are calculated with Vector
 * operations with the outputs of the update and reset gates, and the the
 * outputs from the previous timesteps.
 * 
 * 
 * @author Caleb Devon<br>
 * Created on 5/2/2023
 */

public class GRU_OutputLayer extends RecurrentLayer {
	
	Vector updateVals;
	Vector intermediateVals;
	
	/**
	 * Constructor for the GRU_Output Layer<br>
	 * Activation Function and Bias are pre-determined to be Linear and true
	 * respectively.
	 * @param layerSize Number of Neurons in the layer
	 * @param memoryLength Number of epochs for each training step
	 */
	protected GRU_OutputLayer(int layerSize, int memoryLength) {
		//Activation is Linear due to the lack of any input ConnectionLayers
		super(layerSize, memoryLength, Activation.LINEAR, true);
		updateVals = new Vector(layerSize, Vector.FILL_ZERO);
		intermediateVals = new Vector(layerSize, Vector.FILL_ZERO);
		
	}
	
	/**
	 * Function to set the intermediate values which will be used for calculating the
	 * activation of this layer
	 * @param updateVals result of the update gate
	 * @param intermediateVals result of the intermediate layer
	 */
	protected void setInterVals(Vector updateVals, Vector intermediateVals) {
		this.updateVals = new Vector(updateVals.getVector());
		this.intermediateVals = new Vector(intermediateVals.getVector());
	}

	@Override
	public void runActivation() {
		//Instead of involving any matrix multiplication with a connection layer,
		//the outputlayer output is calculated through 2 consecutive vector operations
		//defined by the GRU algorithm
		
		Vector operationVec = activations.getLast();
		operationVec = Vector.linearMultiply(operationVec, removeBias(updateVals)).getAsVector();
		operationVec = Vector.add(operationVec, removeBias(intermediateVals));
		
		this.pushValues(operationVec);
		this.activate();
		
	}
	
	/**
	 * Recurrent errors for the GRU_OutputLayer is calculated much differently than standard
	 * recurrent layers because there are multiple layers in which this layer connects to that
	 * pass errors back to this one.
	 * @param errorVecs Vector of the error values for each of the destination layers of this 
	 * output layer
	 * @param outputVecs Vector of each of the activation values of each of the destination
	 * layers of this output layer
	 * @param errorMats Matrix of each of the destination RecurrentConnectionLayers that are
	 * sourced from this output layer (only applicable to Reset, Update)
	 */
	protected void calcRecErrors(Vector[] errorVecs, Vector[] outputVecs, Matrix[] errorMats) {
		//ErrorVec/Mat are ordered Reset, Update, Intermediate, Future
		//There is no Future ErrorMat
		//Output is ordered Reset, Update
		
		for(int i = 0; i < errorMats.length; i++) {
			errorMats[i] = Matrix.transpose(errorMats[i]);
		}
		
		//Error calculation has several steps but all are a simple matrix operations
		
		Vector resetE = super.removeBias(Matrix.multiply(errorMats[0], errorVecs[0]).getAsVector());
		Vector updateE = super.removeBias(Matrix.multiply(errorMats[1], errorVecs[1]).getAsVector());
		
		Vector InterE = super.removeBias(Matrix.multiply(errorMats[2], errorVecs[2]).getAsVector());
		InterE = Vector.linearMultiply(InterE, super.removeBias(outputVecs[0])).getAsVector();
		
		Vector one = new Vector(layerSize, Vector.FILL_ONE);
		Vector inverse = Vector.scale(super.removeBias(outputVecs[1]), -1);
		inverse = Vector.add(one, inverse);
		
		Vector futureE = Vector.linearMultiply(inverse, errorVecs[3]).getAsVector();
		
		Vector err1 = Vector.add(resetE, updateE);
		Vector err2 = Vector.add(InterE, futureE);
		
		super.addErrors(Vector.add(err1, err2));
	}
	
	@Override
	protected Vector getValues(int index) {
		return super.getValues(index);
	}
	
	@Override
	protected Vector getRecentValues() {
		return super.getRecentValues();
	}
	
}
