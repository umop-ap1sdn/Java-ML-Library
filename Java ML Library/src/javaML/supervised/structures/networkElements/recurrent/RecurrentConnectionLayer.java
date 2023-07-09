package javaML.supervised.structures.networkElements.recurrent;

import javaML.supervised.structures.Matrix;
import javaML.supervised.structures.networkElements.ConnectionLayer;
import javaML.supervised.structures.networkElements.ffLayerTypes.HiddenLayer;

/**
 * The RecurrentConnectionLayer extension of the ConnectionLayer class is representative of Connections which lie
 * between a single layer or unit between 2 adjacent points in time.<br>
 * They allow values computed in a previous iteration of the Neural Network to have influence over the activations
 * in the current timestep. This will be useful for datasets involving time-series forecasting in which data
 * tends to have some consistency or pattern over time.
 * 
 * @author Caleb Devon<br>
 * Created on 10/14/2022
 *
 */

public class RecurrentConnectionLayer extends ConnectionLayer{
	
	final RecurrentLayer rLayerSrc;
	final HiddenLayer rLayerDest;
	
	/**
	 * Constructor for simple RecurrentLayers in which the ConnectionLayer will lead to and from the same basic
	 * layer, however will operate on 2 separate timesteps.
	 * @param layer RecurrentLayer for which to be connected on
	 */
	public RecurrentConnectionLayer(RecurrentLayer layer) {
		super(layer);
		this.rLayerSrc = layer;
		this.rLayerDest = layer;
	}
	
	/**
	 * Constructor for a more specialized RecurrentConnectionLayer typically found in advanced recurrent models
	 * such as GRUs or LSTMs
	 * @param src RecurrentLayer source will always be necessary
	 * @param dest destination can be any type of HiddenLayer
	 */
	public RecurrentConnectionLayer(RecurrentLayer src, HiddenLayer dest) {
		super(src, dest);
		this.rLayerSrc = src;
		this.rLayerDest = dest;
	}
	
	@Override
	public void adjustWeights(double lr) {
		Matrix gradients = new Matrix(layer.getRows(), layer.getColumns(), Matrix.FILL_ZERO);
		
		for(int row = 0; row < destSize; row++) {
			for(int col = 0; col < sourceSize; col++) {
				double gradient = 0;
				for(int count = 0; count < rLayerSrc.getMemoryLength() - 1; count++) {
					
					/*
					 * A Very minor difference occurs in the RecurrentConnectionLayer adjustWeights() function
					 * Because there are (MemoryLength) timesteps, there will be (MemoryLength - 1) passes between
					 * timesteps, meaning there will be 1 less iteration to adjust weights on
					 */
					
					gradient += rLayerSrc.getValues(count).getValue(col) * 
							rLayerDest.getErrors(count + 1).getValue(row);
				}
				
				gradient *= -lr;
				gradients.setValue(gradient, row, col);
			}
		}
		
		layer = Matrix.add(layer, gradients);
	}
}
