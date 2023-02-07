package javaML.supervised;

/**
 * Normalize is a simple enum file consisting of constants the supported normalization functions
 * within the Network.<br><br>
 * The purpose of normalization is to bring data points of any magnitude to a range more easily "digestible"
 * by a Neural Network/<br><br>
 * The current normalize functions available are None, Sigmoid and Tanh.<br>
 * None does not normalize the data.<br>
 * Sigmoid brings the data to a range of 0 and 1 (much like the range of the sigmoid function).<br>
 * Tanh brings the data to a range of -1 and 1 (much like the range of the tanh function).<br>
 * 
 * @author Caleb Devon<br>
 * Added on 2/6/2023
 *
 */

public enum Normalize {
	TANH_NORMALIZE(11),
	SIGMOID_NORMALIZE(10),
	NONE_NORMALIZE(9),
	INVALID(0);
	


	
	
	
	/*****************************************
	 * Remainder of this file is dedicated to
	 * translating between integer values and
	 * enum constants
	 ****************************************/
	
	
	
	
	
	
	private final int value;
	private Normalize(int value) {
		this.value = value;
	}
	
	public int getVal() {
		return value;
	}
	
	protected static Normalize getFromVal(int value) {
		switch(value) {
		case 9:
			return NONE_NORMALIZE;
		case 10:
			return SIGMOID_NORMALIZE;
		case 11:
			return TANH_NORMALIZE;
		default:
			return INVALID;
		}
	}
	
	@Override
	public String toString() {
		return "" + value;
	}
}