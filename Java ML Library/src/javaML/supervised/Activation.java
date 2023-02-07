package javaML.supervised;

/**
 * Activation is a simple enum file consisting of constants of the various supported activation functions
 * by the Network.<br><br>
 * The current list of supported activation functions are: Linear, ReLU, Sigmoid, and Tanh.
 * 
 * @author Caleb Devon<br>
 * Added on 2/6/2023
 *
 */

public enum Activation {
	INVALID(0), // Invalid, do not use
	LINEAR(1), // Linear activation y = x
	RELU(2), // ReLU activation {x > 0: y = x; x <= 0: y = 0}
	TANH(3), // Tanh activation y = tanh(x)
	SIGMOID(4); // Sigmoid activation y = 1 / (1 + e^(-x))
	
	
	
	
	/*****************************************
	 * Remainder of this file is dedicated to
	 * translating between integer values and
	 * enum constants
	 ****************************************/
	
	
	
	
	private final int value;
	private Activation(int value) {
		this.value = value;
	}
	
	public int getVal() {
		return value;
	}
	
	protected static Activation getFromVal(int value) {
		switch(value) {
		case 1:
			return LINEAR;
		case 2:
			return RELU;
		case 3:
			return TANH;
		case 4:
			return SIGMOID;
		default:
			return INVALID;
		}
	}
	
	@Override
	public String toString() {
		return "" + value;
	}
}