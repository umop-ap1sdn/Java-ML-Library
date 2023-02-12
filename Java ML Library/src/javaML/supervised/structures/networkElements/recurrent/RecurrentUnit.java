package javaML.supervised.structures.networkElements.recurrent;

import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.Vector;
import javaML.supervised.structures.networkElements.ConnectionLayer;
import javaML.supervised.structures.networkElements.NeuronLayer;
import javaML.supervised.structures.networkElements.Unit;

/**
 * The RecurrentUnit extension of the Unit class is similar to the HiddenUnit class, however modified for
 * functionality with RecurrentLayers rather than HiddenLayers. In addition, since the RecurrentLayers
 * have RecurrentConnectionLayers, the RecurrentUnit also includes this feature as well.<br>
 * Much like the HiddenUnit, the RecurrentUnit is also responsible for managing all aspects of the layer.
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class RecurrentUnit extends Unit {
	
	private final RecurrentLayer layer;
	private final RecurrentConnectionLayer rLayer;
	
	/**
	 * Constructor for the RecurrentUnit includes all 3 major components including the RecurrentLayer, 
	 * Ordinary ConnectionLayer, and RecurrentConnectionLayer
	 * @param nLayer RecurrentLayer
	 * @param cLayer ConnectionLayer
	 * @param rLayer RecurrentConnectionLayer
	 */
	public RecurrentUnit(RecurrentLayer nLayer, ConnectionLayer cLayer, RecurrentConnectionLayer rLayer) {
		super(new NeuronLayer[] {nLayer}, new ConnectionLayer[] {cLayer, rLayer});
		this.layer = nLayer;
		this.rLayer = rLayer;
	}

	@Override
	public void forwardPass() {
		this.cLayers[0].forwardPass();
		this.cLayers[1].forwardPass();
	}

	@Override
	public void calcErrors(Unit next, int memIndex) {
		Vector errorVec = next.getEntry().getErrors(memIndex);
		Matrix errorMat = next.getEntryConnections().getMatrix();
		
		layer.calculateErrors(errorVec, errorMat, memIndex);
		
		//Calculate additional correction errors of the previous timestep
		layer.calcRecErrors(rLayer.getMatrix(), memIndex);
		
		layer.putErrors(memIndex);
		
	}
	
}
