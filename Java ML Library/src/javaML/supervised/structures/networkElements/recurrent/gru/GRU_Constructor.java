package javaML.supervised.structures.networkElements.recurrent.gru;

import javaML.supervised.structures.networkElements.ConnectionLayer;
import javaML.supervised.structures.networkElements.NeuronLayer;
import javaML.supervised.structures.networkElements.ffLayerTypes.HiddenLayer;
import javaML.supervised.structures.networkElements.ffLayerTypes.InputLayer;
import javaML.supervised.structures.networkElements.recurrent.RecurrentConnectionLayer;
import javaML.supervised.structures.networkElements.recurrent.RecurrentLayer;

/*******
 * The GRU_Constructor class is designed to make the creation of GRU_Units simply and easily, similar to
 * the NetworkBuilder with the Network class.<br>
 * This class handles the construction of the internals of the GRU including Neuron and Connection layers.
 * 
 * @author Caleb Devon<br>
 * Created on 5/2/2023
 *
 */

public class GRU_Constructor {
	
	/**
	 * Private constructor to enforce non-instantiability
	 */
	private GRU_Constructor() { }
	
	/**
	 * Primary and only function to be called by the NetworkBuilder class whenever a user has indicated a GRU layer
	 * to be used.<br>
	 * Able to complete all procedures of building a GRU Layer
	 * @param previousLayer Neuron Layer which will connect directly into the GRU (important for determining how
	 * the entry ConnectionLayer will be constructed)
	 * @param layerSize Layer size to be used by all internal GRU NeuronLayers - in the future this may be changed
	 * to allow each unique layer its own size
	 * @param memoryLength memory length of the network
	 * @return Returns the constructed GRU_Unit to add the the network layer stack
	 */
	public static GRU_Unit construct(NeuronLayer previousLayer, int layerSize, int memoryLength) {
		HiddenLayer[] nLayers = constructNLayers(layerSize, memoryLength);
		ConnectionLayer[] cLayers = constructCLayers(nLayers, previousLayer);
		
		return new GRU_Unit(nLayers, cLayers);
	}
	
	/**
	 * Private function to be called by the native construct function.<br>
	 * Creates the 4 internal layers which makeup the GRU
	 * @param layerSize Layer size to be used by all internal GRU NeuronLayers - in the future this may be changed
	 * to allow each unique layer its own size 
	 * @param memoryLength memory length of the network
	 * @return Returns the array of NeuronLayers inside the GRU
	 */
	private static HiddenLayer[] constructNLayers(int layerSize, int memoryLength) {
		
		/*
		 * The 4 GRU NeuronLayers are
		 * 
		 * Reset Gate
		 * Update Gate
		 * Intermediate Layer
		 * Output Layer
		 */
		
		ResetLayer resetLayer = new ResetLayer(layerSize, memoryLength);
		UpdateLayer updateLayer = new UpdateLayer(layerSize, memoryLength);
		IntermediateLayer intermediateLayer = new IntermediateLayer(layerSize, memoryLength);
		GRU_OutputLayer output = new GRU_OutputLayer(layerSize, memoryLength);
		
		return new HiddenLayer[] {resetLayer, updateLayer, intermediateLayer, output};
	}
	
	/**
	 * Private function to be called by the native construct function.<br>
	 * Creates the 6 internal connection layers within the GRU model
	 * @param nLayers Array of NeuronLayers which will serve as the endpoints for the ConnectionLayers
	 * @param previousLayer entry layer to be connected to the GRU layers
	 * @return Returns the array of ConnectionLayers inside the GRU
	 */
	private static ConnectionLayer[] constructCLayers(HiddenLayer[] nLayers, NeuronLayer previousLayer) {
		InputLayer input = null;
		HiddenLayer hidden = null;

		GRU_InternalLayer reset = (GRU_InternalLayer) nLayers[0];
		RecurrentLayer output = (RecurrentLayer) nLayers[3];
		
		if(previousLayer instanceof InputLayer) input = (InputLayer) previousLayer;
		else if(previousLayer instanceof HiddenLayer) hidden = (HiddenLayer) previousLayer;
		
		/**
		 * The 6 GRU ConnectionLayers are
		 * 
		 * ixr - input to reset
		 * ixu - input to update
		 * ixin - input to intermediate
		 * hxr - hidden (previous timestep) to reset
		 * hxu - hidden (previous timestep) to update
		 * rxin - reset to intermediate
		 * 
		 * There is no ConnectionLayer from intermediate to output because the output is determined on 
		 * basic vector additions and point-wise multiplications with other pre-calculated vectors
		 */
		
		ConnectionLayer ixr, ixu, ixin, hxr, hxu, rxin;
		
		if(input == null) {
			ixr = new ConnectionLayer(hidden, reset);
			ixu = new ConnectionLayer(hidden, nLayers[1]);
			ixin = new ConnectionLayer(hidden, nLayers[2]);
		} else {
			ixr = new ConnectionLayer(input, reset);
			ixu = new ConnectionLayer(input, nLayers[1]);
			ixin = new ConnectionLayer(input, nLayers[2]);
		}
		
		hxr = new RecurrentConnectionLayer(output, nLayers[0]);
		hxu = new RecurrentConnectionLayer(output, nLayers[1]);
		rxin = new GRU_InternalConnectionLayer(reset, nLayers[2]);
		
		return new ConnectionLayer[] {ixr, hxr, ixu, hxu, ixin, rxin};
	}
}
