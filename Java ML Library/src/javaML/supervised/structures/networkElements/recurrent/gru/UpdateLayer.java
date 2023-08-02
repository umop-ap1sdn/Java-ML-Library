package javaML.supervised.structures.networkElements.recurrent.gru;

import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;

/**
 * The Update Layer is one of the initial layers to the GRU Cell.<br>
 * It is responsible mostly for determining which pieces of data are
 * important to store for long term memory, and one of the reasons it 
 * is said to have solved the vanishing gradient problems of vanilla 
 * recurrent neural networks.
 * 
 * 
 * @author Caleb Devon<br>
 * Created on 5/2/2023
 */

public class UpdateLayer extends GRU_InternalLayer {
	
	private Vector intermediate;
	private Vector previous;
	
	/**
	 * Constructor for the Intermediate Layer<br>
	 * Activation Function and Bias are pre-determined to be TANH and true
	 * respectively.
	 * @param layerSize Number of Neurons in the layer
	 * @param memoryLength Number of epochs for each training step
	 */
	protected UpdateLayer(int layerSize, int memoryLength) {
		super(layerSize, memoryLength);
		initialize();
	}
	
	/**
	 * Because further calculations are done at each layer, additional
	 * error calculations also need to be done using data not native to this
	 * layer.<br>
	 * For the update layer, its error vectors are the output vector of the
	 * intermediate layer and the output vector of the output layer of the 
	 * previous timestep.
	 * @param intermediate first auxiliary vector containing values for error calculation
	 * @param previous second auxiliary vector containing values for error calculation
	 * 
	 */
	protected void setErrorVectors(Vector intermediate, Vector previous) {
		this.intermediate = new Vector(intermediate.getVector());
		this.previous = new Vector(previous.getVector());
	}

	@Override
	public void calculateErrors(Vector errorVec, Matrix errorMat, int memIndex) {
		Vector invert = Vector.scale(previous, -1);
		Vector mult = Vector.add(intermediate, invert);
		
		mult = super.removeBias(mult);
		
		Vector errors = Vector.linearMultiply(errorVec, mult).getAsVector();
		errors = Vector.linearMultiply(errors, this.getDerivatives(memIndex)).getAsVector();
		
		super.addErrors(errors);
	}

	@Override
	public void runActivation() {
		super.activate();
		
		Vector intermediate = new Vector(layerSize, Vector.FILL_ONE);
		Vector sub = Vector.scale(this.activations.getLast(), -1);
		intermediate = Vector.add(intermediate, sub);
		intermediateVals.addLast(intermediate);
		intermediateVals.pollFirst();
		
	}
	
	@Override
	protected Vector getRecentValues() {
		return super.getRecentValues();
	}

}
