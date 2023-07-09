package javaML.supervised.structures.networkElements.recurrent.gru;

import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;
import javaML.supervised.structures.networkElements.ConnectionLayer;
import javaML.supervised.structures.networkElements.NeuronLayer;
import javaML.supervised.structures.networkElements.Unit;

/**
 * The GRU or Gated Recurrent Unit is a specialized type of Neural Network structure.<br>
 * Similar to the RNN, Recurrent Neural Network, it is most often used for sequential data such as Stock Prices,
 * or Language Processing.<br>
 * The GRU was created as a solution to the RNN's most documented issue, the Vanishing Gradient Problem, which
 * causes the RNN to have poor memory for long sequences. The GRU is able to decrease the severity of this issue
 * through the use of Gates - specialized Neural Network Layers that learn to recognize data as important and 
 * non-important, allowing a long term memory to be created.
 * 
 * 
 * @author Caleb Devon<br>
 * Created on 5/2/2023
 *
 */

public class GRU_Unit extends Unit{
	
	
	//Neuron Layers contained within the Unit
	ResetLayer reset;
	UpdateLayer update;
	IntermediateLayer intermediate;
	GRU_OutputLayer output;
	
	//Index values for each of the ConnectionLayers
	private static final int IXR = 0, HXR = 1, IXU = 2, HXU = 3, IXIN = 4, RXIN = 5;
	
	/**
	 * Constructor for the GRU_Unit<br>
	 * To be called only by the GRU_Constructor which is able to create the Arrays for the nLayers and cLayers
	 * @param nLayers Array of NeuronLayers in the GRU
	 * @param cLayers Array of ConnectionLayers in the GRU
	 */
	protected GRU_Unit(NeuronLayer[] nLayers, ConnectionLayer[] cLayers) {
		super(nLayers, cLayers);
		
		this.reset = (ResetLayer) nLayers[0];
		this.update = (UpdateLayer) nLayers[1];
		this.intermediate = (IntermediateLayer) nLayers[2];
		this.output = (GRU_OutputLayer) nLayers[3];
	}

	@Override
	public void forwardPass() {
		
		//Many of the internal layers rely on values to be already activated by other internal layers before
		//performing their forward pass
		//For this reason the GRU_Unit must break tradition and activate the internal layers before the main
		//process of forward passing has finished
		
		cLayers[IXR].forwardPass();
		cLayers[HXR].forwardPass();
		reset.setHiddenState(output.getRecentValues());
		reset.runActivation();
		
		cLayers[IXU].forwardPass();
		cLayers[HXU].forwardPass();
		update.runActivation();
		
		cLayers[IXIN].forwardPass();
		cLayers[RXIN].forwardPass();
		intermediate.setHiddenState(update.getRecentValues());
		intermediate.runActivation();
	}
	
	@Override
	public void runActivation() {
		output.setInterVals(update.getRecentIntermediateVals(), intermediate.getRecentIntermediateVals());
		output.runActivation();
	}

	@Override
	public void calcErrors(Unit next, int memIndex) {
		
		//A higher involved process is needed to pass the errors through the GRU cell
		
		Vector errorVec = next.getEntryErrors(memIndex);
		Matrix errorMat = next.getEntryMatrix();
		
		//Create lists of recurrent pass-back errors
		Vector[] errorVecs = {reset.getErrors(memIndex + 1), update.getErrors(memIndex + 1), 
				intermediate.getErrors(memIndex + 1), output.getErrors(memIndex + 1)
		};
		
		Vector[] outputVecs = {reset.getValues(memIndex + 1), update.getValues(memIndex + 1)};
		Matrix[] errorMats = {cLayers[HXR].getMatrix(), cLayers[HXU].getMatrix(), cLayers[RXIN].getMatrix()};
		
		//Error calculation starts at the output layer
		output.calculateErrors(errorVec, errorMat, memIndex);
		output.calcRecErrors(errorVecs, outputVecs, errorMats);
		output.putErrors(memIndex);
		
		//Passes back to the intermediate layer
		//Many of the internal layers need 1 or 2 extra vectors to be sent to complete error pass
		intermediate.setErrorVector(update.getValues(memIndex));
		intermediate.calculateErrors(output.getErrors(memIndex), null, memIndex);
		intermediate.putErrors(memIndex);
		
		//Update and Reset layer errors can be calculated in either order
		
		update.setErrorVectors(intermediate.getValues(memIndex), output.getValues(memIndex - 1));
		update.calculateErrors(output.getErrors(memIndex), null, memIndex);
		update.putErrors(memIndex);
		
		reset.setErrorVector(output.getValues(memIndex - 1));
		reset.calculateErrors(intermediate.getErrors(memIndex), cLayers[RXIN].getMatrix(), memIndex);
		reset.putErrors(memIndex);
	}
	
	@Override
	public Vector getEntryErrors(int memIndex) {
		
		//Because the function getEntryErrors(int) expects only 1 vector to be sent
		//the error pass-back algorithm is calculated right here instead of in previous layer
		
		//NOTE: Might be bad practice, but this was the best way I found to do this without replacing a lot of code
		
		Vector error1 = Matrix.multiply(Matrix.transpose(cLayers[IXR].getMatrix()), 
				reset.getErrors(memIndex)).getAsVector();
		Vector error2 = Matrix.multiply(Matrix.transpose(cLayers[IXU].getMatrix()), 
				update.getErrors(memIndex)).getAsVector();
		Vector error3 = Matrix.multiply(Matrix.transpose(cLayers[IXIN].getMatrix()), 
				intermediate.getErrors(memIndex)).getAsVector();
		
		Vector errors = Vector.add(error1, error2);
		errors = Vector.add(errors, error3);
		
		return errors;
		
	}
	
	@Override
	public Matrix getEntryMatrix() {
		//int layerSize = cLayers[IXR].getMatrix().getColumns() - (cLayers[IXR].getSource().getBias() ? 1 : 0);
		
		//Similar to the EntryErrors dirty fix, since the pass-back errors are pre-calculated the entry matrix
		//is modified to be the identity matrix of size equal to layersize
		
		//Ultimately, when passed back, the previous layer to this will calculate errors using the pre-calculated
		//vector multiplied by this matrix (transposed) multiplied by its own inputs
		//Since I_t = I, this formula becomes Error = E_vec * I * Prev_Outputs or just simply
		//Error = E_vec * Prev_Outputs
		
		//Once again a dirty fix that I used to not have to modify a great deal of code
		
		int layerSize = cLayers[IXR].getMatrix().getColumns();
		return new Matrix(layerSize, layerSize, Matrix.FILL_IDENTITY);
	}

}
