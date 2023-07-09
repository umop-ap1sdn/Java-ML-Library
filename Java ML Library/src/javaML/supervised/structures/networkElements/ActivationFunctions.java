package javaML.supervised.structures.networkElements;


/**
 * A collection of purely math classes for Neural Network activation functions<br>
 * Activation Functions are used by Neural Networks for several reasons, based on the nature of the function
 * used.
 * Generally, activation functions are designed to be non-linear transformations, and often restrict
 * the range.
 * Activations are applied at the end of a forward pass step when the matrix multiplication has completed.
 * 
 * @author Caleb Devon<br>
 * Created 10/14/2022
 *
 */
public abstract class ActivationFunctions {
	
	/**
	 * Primary function for the activating of inputs
	 * @param x Input
	 * @return Activated output
	 */
	protected abstract double activate(double x);
	
	/**
	 * Function to get the derivative (slope) of the activation function at a particular x<br>
	 * This will be used for backpropagation, and is a core aspect to the gradient descent algorithm
	 * @param x Original, unactivated input
	 * @return Derivative/slope of the function at the particular x
	 */
	protected abstract double derivative(double x);
	
	/**
	 * Shortcut function to activate an entire input vector
	 * @param f Activation function to be used
	 * @param x array of input values
	 * @return Array of activated values from the inputs
	 */
	protected static double[] activate(ActivationFunctions f, double[] x) {
		double[] ret = new double[x.length];
		for(int index = 0; index < x.length; index++) {
			ret[index] = f.activate(x[index]);
		}
		
		return ret;
	}
	
	/**
	 * Shortcut function to derive an entire input vector
	 * @param f Activation function derivative to be used
	 * @param x array of input values
	 * @return Array of derived values from the inputs
	 */
	protected static double[] derivative(ActivationFunctions f, double[] x) {
		double[] ret = new double[x.length];
		for(int index = 0; index < x.length; index++) {
			ret[index] = f.derivative(x[index]);
		}
		
		return ret;
	}
}

/**
 * The Linear function is the most basic activation function and is typically used only for the input
 * layer.<br><br>
 * Its activation consists of a y = x equation and its derivative is y = 1<br>
 * The linear function has no restricted range
 * @author Caleb Devon
 *
 */
class Linear extends ActivationFunctions {
	
	@Override
	protected double activate(double x) {
		return x;
	}
	
	@Override
	protected double derivative(double x) {
		return 1;
	}
}

/**
 * The ReLU function is a similarly basic function but changes a major aspect to ensure its nonlinearity.
 * It is most often used for the hidden layers of feed forward neural networks thanks to their simplicity.
 * <br><br>
 * Its activation is a piece-wise function {x <= 0: y = 0, x > 0: y = x}.
 * As such its derivative is also piece-wise {x <= 0: y = 0, x > 0: y = 1}<br>
 * This function has a restricted range of [0, Infinity)
 * @author Caleb Devon
 *
 */
class ReLU extends ActivationFunctions {
	
	@Override
	protected double activate(double x) {
		if(x > 0) return x;
		return 0;
	}
	
	@Override
	protected double derivative(double x) {
		if(x > 0) return 1;
		return 0;
	}
}

/**
 * The Tanh function is fairly self-explanatory in that it is the hyperbolic tangent function.
 * It can be used for any layer of a network with exception to the input layer (which is always linear),
 * but is computationally expensive.<br><br>
 * Its activation function is y = tanh(x) and its derivative is y = sech^2(x)<br>
 * This function has a restricted range of (-1, 1)
 * @author Caleb Devon
 *
 */
class Tanh extends ActivationFunctions {
	
	@Override
	protected double activate(double x) {
		return Math.tanh(x);
	}
	
	@Override
	protected double derivative(double x) {
		return 1 / Math.pow(Math.cosh(x), 2);
	}
}

/**
 * The Sigmoid function is likely the most commonly used activation function for machine learning mostly
 * due to its range (0, 1). In many aspects, it has a very similar shape to the tanh function with the main 
 * difference being its mentioned range. As such the sigmoid function is often used in 
 * similar circumstances to the tanh function, in which having a (0, 1) range is more conducive 
 * than a (-1, 1) range.<br><br>
 * Its activation function is y = 1 / (1 + e^(-x)) and its derivative is sigmoid(x)(1 - sigmoid(x))<br>
 * This function has a restricted range of (0, 1)
 * @author Caleb Devon
 *
 */
class Sigmoid extends ActivationFunctions {
	
	@Override
	protected double activate(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	@Override
	protected double derivative(double x) {
		return this.activate(x) * (1 - this.activate(x));
	}
}