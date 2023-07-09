package javaML.supervised.structures.networkElements.ffLayerTypes;

import javaML.supervised.Activation;
import javaML.supervised.LayerType;
import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;
import javaML.supervised.structures.networkElements.NeuronLayer;

/**
 * The InputLayer extension of the NeuronLayer class is representative of the entry-point to the Neural Network.
 * <br><br>
 * The key properties of the InputLayer include:<br>
 * - Its values are User-Defined; this means their values are not the result of some prior calculation.<br>
 * - They lack a Non-Linear activation function; unlike the standard for hidden and output layers, since the input
 * layer values are not the result of matrix multiplication, they have no need to be activated by a standard, 
 * non linear function, and therefore are given the linear function.<br>
 * - User can decide whether or not to include a bias node.
 * <br><br>
 * 
 * Each Neural Network will have exactly 1 InputLayer.
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class InputLayer extends NeuronLayer {
	
	/**
	 * Constructor for the InputLayer extension of the NeuronLayer class.<br>
	 * Constructor is nearly identical to its parent, but removes the ActivationCode parameter as the input layer is
	 * locked into a Linear activation function
	 * @param layerSize Length of the vector
	 * @param memoryLength Number of time steps in which to save past data
	 * @param bias Boolean for whether or not to include a bias in the values vector
	 */
	public InputLayer(int layerSize, int memoryLength, boolean bias) {
		super(layerSize, memoryLength, Activation.LINEAR, bias);
	}
	
	@Override
	public void runActivation() {
		//Nothing special because there are no input matrix to an input layer
		this.activate();
	}
	
	@Override
	protected void calculateErrors(Vector errorVec, Matrix errorMat, int memIndex) {
		//By Definition the input layer does not have an error
		Vector error = new Vector(this.getLayerSize(), Matrix.FILL_ZERO);
		error = super.removeBias(error);
		super.addErrors(error);
		super.putErrors(memIndex);
	}
	
	/**
	 * Function to set the values of the input layer, which will serve as the input vector for the network.
	 * @param inputs 1D array for the vector to set the input layer
	 */
	public void setInputs(double[] inputs) {
		setInputs(new Vector(inputs));
	}
	
	/**
	 * Overload function for setting inputs with added functionality for Vector as the parameter.
	 * @param inputs Vector for which to set the input layer values to.
	 */
	private void setInputs(Vector inputs) {
		this.pushValues(inputs);
	}

	@Override
	public String toString() {
		return String.format("%s,%d,%s,%d\n", LayerType.INPUT, layerSize, activationCode, bias ? 1 : 0);
	}
	
}
