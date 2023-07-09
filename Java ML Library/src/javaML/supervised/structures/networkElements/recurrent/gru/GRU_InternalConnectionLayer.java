package javaML.supervised.structures.networkElements.recurrent.gru;

import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;
import javaML.supervised.structures.networkElements.ConnectionLayer;
import javaML.supervised.structures.networkElements.ffLayerTypes.HiddenLayer;

/**
 * The GRU_InternalConnectionLayer is a ConnectionLayer specialized for the GRU Cell.<br>
 * Its primary difference from standard ConnectionLayers is that the source values it
 * uses for forward propagation are the intermediate values from the GRU NeuronLayer
 * rather than the basic outputs from other ConnectionLayer types.
 * 
 * 
 * @author Caleb Devon<br>
 * Created on 5/2/2023
 *
 */

public class GRU_InternalConnectionLayer extends ConnectionLayer {
	
	GRU_InternalLayer source;
	
	/**
	 * Constructor for the GRU_InternalConnectionLayer.<br>
	 * Source must be a GRU_InternalLayer however the Destination can be any HiddenLayer
	 * variant (such as the GRU_OutputLayer which is categorized as Recurrent).
	 * 
	 * @param source Source NeuronLayer (Must be GRU_InternalLayer)
	 * @param destination Destination NeuronLayer (Can be any HiddenLayer)
	 */
	public GRU_InternalConnectionLayer(GRU_InternalLayer source, HiddenLayer destination) {
		super(source, destination);
		this.source = source;
	}
	
	@Override
	public void forwardPass() {
		//Retrieve Intermediate vals
		//Intermediate vals refer to process that occur after the initial forward propagation
		Vector result = Matrix.multiply(layer, source.getRecentIntermediateVals()).getAsVector();
		destination.pushValues(result);
	}

}
