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
	TANH_NORMALIZE,
	SIGMOID_NORMALIZE,
	NONE_NORMALIZE,
	INVALID;
}