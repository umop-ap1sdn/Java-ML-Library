package javaML.supervised;

/**
 * LayerType is a simple enum file consisting of constants of the various supported layer types
 * within the Network.<br><br>
 * The current list of supported layer types are: Input, Hidden, Recurrent, Output.
 * 
 * @author Caleb Devon<br>
 * Added on 2/6/2023
 *
 */

public enum LayerType {
	INVALID(0), 	// Invalid, Do not use
	INPUT(5), 		// Input Layer
	HIDDEN(6), 		// Hidden Layer
	RECURRENT(7), 	// Recurrent Layer
	OUTPUT(8); 		// Output Layer
	

	
	
	
	/*****************************************
	 * Remainder of this file is dedicated to
	 * translating between integer values and
	 * enum constants
	 ****************************************/
	
	
	
	
	
	private final int value;
	private LayerType(int value) {
		this.value = value;
	}
	
	public int getVal() {
		return value;
	}
	
	protected static LayerType getFromVal(int value) {
		switch(value) {
		case 5:
			return INPUT;
		case 6:
			return HIDDEN;
		case 7:
			return RECURRENT;
		case 8:
			return OUTPUT;
		default:
			return INVALID;
		}
	}
	
	@Override
	public String toString() {
		return "" + value;
	}
}