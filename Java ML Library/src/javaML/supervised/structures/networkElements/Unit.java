package javaML.supervised.structures.networkElements;

import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;

/**
 * The Unit class is an abstract structure designed to manage all parts of any particular layer.
 * As such it contains both the Neuron Layers and Connection Layers and is capable of performing 
 * forward pass, layer activation, error calculation, and backpropagation.
 * <br><br>
 * Extensions of this class will be made to create area specific Unit to perform their specific tasks.
 * Some examples of these include the HiddenUnit and OutputUnit.
 * <br><br>
 * Note that any Unit does not require the ConnectionLayer that represents the output of a particular
 * NeuronLayer. This is because for n NeuronLayers, there will be n - 1 ConnectionLayers between them, and as
 * such all ConnectionLayers are properly accounted for without needing to include outputs. Additionally, 
 * this means it is not necessary to include an InputUnit, and instead, Networks are built with the standard
 * InputLayer, before switching to Units for hidden and output layers.
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */
public abstract class Unit {
	
	//Class needs to be able to pass forward, error propagate, and adjust weights
	//May contain any number neuron layers and connection layers
	//a neuron layer needs to support a multiple number of input matrices and output matrices
	//connection layers still need only one source and destination
	
	protected final NeuronLayer[] nLayers;
	protected final ConnectionLayer[] cLayers;
	
	protected final int nLayersSize;
	protected final int cLayersSize;
	
	/**
	 * Constructor for complex Unit. This will be used for network components with multiple neuron
	 * and connection layers.<br>
	 * One example of a complex unit is a recurrent unit which still only has one NeuronLayer but 2
	 * ConnectionLayers, one feed forward layer and one recurrent layer
	 * @param nLayers Array of NeuronLayers (index 0 will be the entry point
	 * @param cLayers Array of ConectionLayers
	 */
	protected Unit(NeuronLayer[] nLayers, ConnectionLayer[] cLayers) {
		this.nLayers = nLayers;
		this.cLayers = cLayers;
		
		this.nLayersSize = nLayers.length;
		this.cLayersSize = cLayers.length;
	}
	
	/**
	 * Constructor for a simple Unit. This represents a Unit with only one NeuronLayer and one
	 * ConnectionLayer, and will be extended to the HiddenUnit and OutputUnit
	 * @param nLayer NeuronLayer for the unit
	 * @param cLayer ConnectionLayer for the unit
	 */
	protected Unit(NeuronLayer nLayer, ConnectionLayer cLayer) {
		nLayers = new NeuronLayer[]{nLayer};
		cLayers = new ConnectionLayer[]{cLayer};
		
		nLayersSize = 1;
		cLayersSize = 1;
	}
	
	/**
	 * Function to be called only by the Network class.<br>
	 * This function is designed to pass forward the values from the previous layer
	 */
	public abstract void forwardPass();
	
	/**
	 * Function to be called only by the Network class.<br>
	 * This function is responsible for calculating the errors of a particular unit.
	 * @param next Next unit is passed as its error typically influences the errors of layers before itself.<br>
	 * OutputUnit will pass a null since it is the final Unit
	 * @param memIndex Location (timestep) in memory from which to calculate error from.
	 */
	public abstract void calcErrors(Unit next, int memIndex);
	
	/**
	 * Function to be called only by the Network class.<br>
	 * This function is responsible for activating the layer after forward pass has been completed.<br>
	 * It is important for this step to occur after forward pass to ensure all passes have occurred.
	 */
	public void runActivation() {
		for(NeuronLayer n: nLayers) n.runActivation();
	}
	
	/**
	 * Function to be called only by the Network class.<br>
	 * This function runs the weight adjustment algorithm for all ConnectionLayers in the network
	 * @param lr Parameter for the learning rate in which to adjust the weights by
	 */
	public void backpropagation(double lr) {
		for(ConnectionLayer c: cLayers) c.adjustWeights(lr);
	}
	
	/**
	 * Function to reset all the NeuronLayers contained in the Unit
	 */
	public void reset() {
		for(NeuronLayer n: nLayers) n.reset();
	}
	
	/**
	 * Function to purge the errors of all NeuronLayers held within a Unit
	 * @param batchSize Network's batch size, is used to determine how far to move the errors.
	 */
	public void purgeErrors(int batchSize) {
		for(NeuronLayer n: nLayers) n.purgeErrors(batchSize);
	}
	
	/**
	 * Function to override all matrices of each ConnectionLayer contained in the Unit
	 * @param matrices 3D array containing a list of 2D arrays to replace the matrices of each
	 * ConnectionLayer
	 */
	public void setConnectionMatrices(double[][][] matrices) {
		for(int index = 0; index < cLayers.length; index++) {
			cLayers[index].setMatrix(matrices[index]);
		}
	}
	
	/**
	 * Function to get the entry point (NeuronLayer) for the Unit
	 * @return Returns the Vector to the first NeuronLayer of the Unit
	 */
	public Vector getEntryErrors(int memIndex) {
		return this.nLayers[0].getErrors(memIndex);
	}
	
	/**
	 * Function to get the entry point (ConnectionLayer) for the Unit
	 * @return Returns the Matrix of the ConnectionLayer which leads into the first NeuronLayer
	 */
	public Matrix getEntryMatrix() {
		return this.cLayers[0].getMatrix();
	}
	
	/**
	 * Function to get the exit point (NeuronLayer) for the Unit
	 * @return Returns the last NeuronLayer of the Unit
	 */
	public NeuronLayer getExit() {
		return this.nLayers[nLayersSize - 1];
	}
	
	@Override
	public String toString() {
		String ret = "";
		
		ret += nLayers[0];
		ret += "-\n";
		for(ConnectionLayer c: cLayers) ret += c + "-\n";
		ret += "--\n";
		
		return ret;
	}
	
}
