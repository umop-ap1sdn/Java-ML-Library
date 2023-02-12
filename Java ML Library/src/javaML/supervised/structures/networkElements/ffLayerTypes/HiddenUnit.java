package javaML.supervised.structures.networkElements.ffLayerTypes;

import javaML.supervised.structures.*;
import javaML.supervised.structures.networkElements.*;

/**
 * The HiddenUnit extension of the Unit class is designed to oversee all aspects of a HiddenLayer.<br>
 * As such it manages any given HiddenLayer to perform the forward pass, error calculation, and backpropagation
 * on its layer.
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class HiddenUnit extends Unit {
	
	private final HiddenLayer layer;
	
	/**
	 * Basic Constructor for HiddenUnit class.<br>
	 * Since a HiddenUnit has only one NeuronLayer and one ConnectionLayer, this is all the constructor asks for
	 * @param layer HiddenLayer
	 * @param conIn Source connection for the HiddenLayer
	 */
	public HiddenUnit(HiddenLayer layer, ConnectionLayer conIn) {
		super(layer, conIn);
		this.layer = layer;
	}

	@Override
	public void forwardPass() {
		this.cLayers[0].forwardPass();
	}

	@Override
	public void calcErrors(Unit next, int memIndex) {
		Vector errorVec = next.getEntry().getErrors(memIndex);
		Matrix errorMat = next.getEntryConnections().getMatrix();
		layer.calculateErrors(errorVec, errorMat, memIndex);
		layer.putErrors(memIndex);
	}
}
