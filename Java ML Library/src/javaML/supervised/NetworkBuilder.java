package javaML.supervised;

import java.io.File;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.LinkedList;
import java.util.Scanner;

import javaML.supervised.structures.networkElements.ConnectionLayer;
import javaML.supervised.structures.networkElements.NeuronLayer;
import javaML.supervised.structures.networkElements.Unit;
import javaML.supervised.structures.networkElements.ffLayerTypes.HiddenLayer;
import javaML.supervised.structures.networkElements.ffLayerTypes.HiddenUnit;
import javaML.supervised.structures.networkElements.ffLayerTypes.InputLayer;
import javaML.supervised.structures.networkElements.ffLayerTypes.OutputLayer;
import javaML.supervised.structures.networkElements.ffLayerTypes.OutputUnit;
import javaML.supervised.structures.networkElements.recurrent.RecurrentConnectionLayer;
import javaML.supervised.structures.networkElements.recurrent.RecurrentLayer;
import javaML.supervised.structures.networkElements.recurrent.RecurrentUnit;
import javaML.supervised.structures.networkElements.recurrent.gru.GRU_Constructor;
import javaML.supervised.structures.networkElements.recurrent.gru.GRU_Unit;

/**
 * NetworkBuilder is a class to be declared by the Client program<br>
 * This class is responsible for creating a network to the specifications of the user<br>
 * The use of this file allows the creation of networks to be simple and readable on Client side
 * @author Caleb Devon<br>
 * created on 10/14/2022
 *
 */
public class NetworkBuilder {
	
	private boolean allowInput, allowHidden, allowOutput, allowFinalize;
	private int memoryLength;
	//private int numRecurrent;
	
	private InputLayer input;
	private OutputUnit output;
	private ArrayList<Unit> hiddenLayers;
	
	/**
	 * Standard constructor to be used in Client side code
	 * @param memoryLength Declared to specify how deep each neuron layer will remember inputs for.<br>
	 * This will be used during backpropagation to determine how many iterations will be run during
	 * a backpropagation routine
	 */
	public NetworkBuilder(int memoryLength) {
		this.memoryLength = memoryLength;
		allowInput = true;
		allowHidden = false;
		allowOutput = false;
		allowFinalize = false;
		
		//numRecurrent = 0;
		
		hiddenLayers = new ArrayList<>();
		
	}
	
	/**
	 * Function to add a new layer to the in progress network
	 * @param layerType Declares what type of layer to be created<br>
	 * Use the LayerType enum to define which layer type to create<br>
	 * Use LayerType.INPUT, LayerType.HIDDEN, LayerType.RECURRENT or LayerType.OUTPUT to declare different layers
	 * @param layerSize Declares the size of the layer not including a bias the user may or may not 
	 * choose to employ
	 * @param activation Declares the activation function to be used by the layer<br>
	 * This parameter is ignored when declaring Input layers since input layers can only use linear 
	 * activations<br>
	 * Use the Activation enum to define which activation function to be used
	 * These constants consist of Activation.LINEAR, Activation.RELU, Activation.SIGMOID, and Activation.TANH
	 * @param bias boolean value for whether to include a bias in the the layer<br>
	 * Bias nodes are nodes which connect to the layer ahead of itself<br>
	 * If the layer type is an output layer this parameter is ignored because biases are irrelevant
	 * for output layers
	 * @return Returns true if the layer was successfully created<br>
	 * In order for layers to be successful it must follow a few rules:<br>
	 * <ul>
	 * <li>First layer MUST be an Input layer</li>
	 * <li>After first layer is created, no more input layers can be created</li>
	 * <li>Network can support as many hidden layers as desired, but no more layers can be added after
	 * after an output layer</li>
	 * <li>After the output layer is created, Network MUST be finalized</li>
	 * </ul>
	 * 
	 */
	public boolean putLayer(LayerType layerType, int layerSize, Activation activation, boolean bias) {
		
		switch(layerType) {
		case INPUT:
			return putInputLayer(layerSize, bias);
		case HIDDEN:
			return putHiddenLayer(layerSize, activation, bias);
		case RECURRENT:
			return putRecurrentLayer(layerSize, activation, bias);
		case OUTPUT:
			return putOutputLayer(layerSize, activation);
		case GRU:
			return putGRULayer(layerSize);
		case INVALID:
			return false;
		}
		
		return false;
	}
	
	/**
	 * Private function that is called by the putLayer() function when the layerType is declared as
	 * an input layer
	 * @param layerSize Size of the layer, not including potential bias
	 * @param bias boolean for whether a bias is included
	 * @return true if the layer was successfully created
	 */
	private boolean putInputLayer(int layerSize, boolean bias) {
		if(!allowInput) return false;
		
		input = new InputLayer(layerSize, memoryLength, bias);
		
		allowHidden = true;
		allowOutput = true;
		allowInput = false;
		
		return true;
	}
	
	/**
	 * Private function that is called by the putLayer() function when the layerType is declared as
	 * a hidden layer
	 * @param layerSize Size of the layer, not including potential bias
	 * @param activation Activation function to be used by the layer
	 * @param bias boolean for whether a bias is included
	 * @return true if the layer was successfully created
	 */
	private boolean putHiddenLayer(int layerSize, Activation activation, boolean bias) {
		if(!allowHidden) return false;
		
		HiddenLayer hidden = new HiddenLayer(layerSize, memoryLength, activation, bias);
		int prev = hiddenLayers.size() - 1;
		
		ConnectionLayer con;
		if(prev == -1) con = new ConnectionLayer(input, hidden);
		else con = new ConnectionLayer((HiddenLayer) hiddenLayers.get(prev).getExit(), hidden);
		
		hiddenLayers.add(new HiddenUnit(hidden, con));
		
		return true;
	}
	
	/**
	 * Private function that is called by the putLayer() function when the layerType is declared as
	 * a recurrent layer
	 * @param layerSize Size of the layer, not including potential bias
	 * @param activation Activation function to be used by the layer
	 * @param bias boolean for whether a bias is included
	 * @return true if the layer was successfully created
	 */
	private boolean putRecurrentLayer(int layerSize, Activation activation, boolean bias) {
		if(!allowHidden) return false;
		
		RecurrentLayer rLayer = new RecurrentLayer(layerSize, memoryLength, activation, bias);
		int prev = hiddenLayers.size() - 1;
		
		ConnectionLayer con;
		if(prev == -1) con = new ConnectionLayer(input, rLayer);
		else con = new ConnectionLayer((HiddenLayer) hiddenLayers.get(prev).getExit(), rLayer);
		
		RecurrentConnectionLayer rCon = new RecurrentConnectionLayer(rLayer);
		
		hiddenLayers.add(new RecurrentUnit(rLayer, con, rCon));
		
		return true;
	}
	
	/**
	 * Private function that is called by the putLayer() function when the layerType is declared as
	 * a GRU layer.<br>
	 * Activation and Bias do not need to be specified because the GRU structure already declares these parameters.
	 * @param layerSize Size of the layer, not including potential bias
	 * @return true if the layer was successfully created
	 */
	private boolean putGRULayer(int layerSize) {
		if(!allowHidden) return false;
		
		GRU_Unit gru;
		int prev = hiddenLayers.size() - 1;
		
		NeuronLayer prevLayer;
		if(prev == -1) prevLayer = input;
		else prevLayer = hiddenLayers.get(prev).getExit();
		
		gru = GRU_Constructor.construct(prevLayer, layerSize, memoryLength);
		
		hiddenLayers.add(gru);
		
		return true;
	}
	
	
	/**
	 * Private function that is called by the putLayer() function when the layerType is declared as
	 * an output layer
	 * @param layerSize Size of the layer, not including potential bias
	 * @param activation Activation function to be used by the layer
	 * @return true if the layer was successfully created
	 */
	private boolean putOutputLayer(int layerSize, Activation activation) {
		if(!allowOutput) return false;
		
		OutputLayer outputLayer = new OutputLayer(layerSize, memoryLength, activation);
		
		ConnectionLayer con;
		int prev = hiddenLayers.size() - 1;
		
		if(prev == -1) con = new ConnectionLayer(input, outputLayer);
		else con = new ConnectionLayer((HiddenLayer) hiddenLayers.get(prev).getExit(), outputLayer);
		
		output = new OutputUnit(outputLayer, con);
		
		allowHidden = false;
		allowOutput = false;
		allowFinalize = true;
		
		return true;
	}
	
	/**
	 * Function to be called in the client side code to finalize the construction of a network
	 * @param batchSize value that will be set as the default for the batch that will be trained anytime
	 * the train() function from the Network class is called 
	 * @return returns the newly built network if the network is valid<br>
	 * If the network is not ready to be finalized (no input/output currently exists) then a null will
	 * be returned
	 * 
	 */
	public Network build(int batchSize) {
		if(!allowFinalize) return null;
		
		Unit[] hidden = new Unit[hiddenLayers.size()];
		
		
		for(int index = 0; index < hidden.length; index++) hidden[index] = hiddenLayers.get(index);
		
		
		return new Network(input, output, hidden, batchSize, memoryLength);
	}
	
	/**
	 * Writes a file for given network<br>
	 * This allows the user to save a trained network for later use
	 * @param network The network to be saved
	 * @param fileName The name of the file to write to (extension = .nn)
	 * @param avoidOverwriting If true, program will add a number to the end of the file name to ensure no
	 * files are overwritten.
	 * @return Returns true if the file was successfully written
	 */
	
	public static boolean writeFile(Network network, String fileName, boolean avoidOverwriting) {
		try {
			File folder = new File("files//networks//");
			if(!folder.exists()) folder.mkdirs();	//Make standard directory for network saving location
			
			String path = String.format("files//networks//%s", fileName);
			String extension = ".nn";
			
			int addition = 0;
			
			File file;
			
			//Add unique # to the end of the file name to ensure a new file will be written
			//(if avoidOverwriting is set to true)
			if(avoidOverwriting) {
				do {
					file = new File(String.format("%s%d%s", path, addition++, extension));
				} while(file.exists());
			} else {
				file = new File(path + extension);
			}
			
			Formatter fileWriter = new Formatter(file);
			
			fileWriter.format("%s", network);	//Implicitly calls the Network.toString() function
			
			fileWriter.close();
			
			//Final check to ensure file creation was successful
			if(file.exists()) return true;
			
		} catch (Exception e) {
			return false;
		}
		
		return false;
	}
	
	/**
	 * Function to build a network from a file, such as one that might be created from the writeFile() 
	 * function
	 * @param path File path to the target file
	 * @return Returns the network built from the file, if one could be made
	 */
	
	public static Network buildFromFile(String path) {
		try {
			File file = new File(path);
			Scanner sc = new Scanner(file);
			
			LinkedList<String> instructions = new LinkedList<>();
			
			//Load all lines of the file into LinkedList
			//Every time a line is read, it is deleted from the list
			//This way the program can continuously read from the head of the linked list instead of 
			//following an index
			while(sc.hasNextLine()) {
				instructions.addLast(sc.nextLine());
			}
			
			sc.close();
			
			//line 1 - memoryLength and batchSize
			String[] start = instructions.pollFirst().split(",");
			NetworkBuilder bob = new NetworkBuilder(Integer.parseInt(start[0]));
			int batchSize = Integer.parseInt(start[1]);
			
			String inputSpecs = instructions.pollFirst();
			buildLayer(bob, inputSpecs);
			instructions.pollFirst();
			buildNetwork(bob, instructions);
			
			return bob.build(batchSize);
			
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		
	}
	
	private static void buildLayer(NetworkBuilder bob, String specs) {
		String[] arr = specs.split(",");
		int layerType = Integer.parseInt(arr[0]);
		int layerSize = Integer.parseInt(arr[1]);
		int activation = Integer.parseInt(arr[2]);
		boolean bias = Integer.parseInt(arr[3]) == 1;
		
		bob.putLayer(LayerType.getFromVal(layerType), layerSize, Activation.getFromVal(activation), bias);
	}
	
	/**
	 * Function to be called only by the Native class<br>
	 * This function uses lines from a file to add network layers
	 * @param bob NetworkBuilder to handle building of layers 
	 * @param instructions Lines from a file loaded into a LinkedList
	 */
	private static void buildNetwork(NetworkBuilder bob, LinkedList<String> instructions) {
		String line = instructions.pollFirst();
		int unitIndex = 0;
		ArrayList<double[][]> cLayers = new ArrayList<>();
		ArrayList<double[]> arrs = new ArrayList<>();
		
		while(!line.equals("---")) {
			buildLayer(bob, line);
			instructions.pollFirst();
			line = instructions.pollFirst();
			
			while(!line.equals("--")) {
				
				if(line.equals("-")) {
					cLayers.add(buildArray(arrs));
					arrs = new ArrayList<>();
					line = instructions.pollFirst();
					continue;
				}
				
				String[] arr = line.split(",");
				arrs.add(StringtoDub(arr));
				
				line = instructions.pollFirst();
			}
			
			if(unitIndex < bob.hiddenLayers.size()) 
				bob.hiddenLayers.get(unitIndex++).setConnectionMatrices(buildCLayers(cLayers));
			else bob.output.setConnectionMatrices(buildCLayers(cLayers));
			
			cLayers = new ArrayList<>();
			line = instructions.pollFirst();
		}
	}
	
	/**
	 * Function to be called only by the native class
	 * Turns a String array into an array of double values
	 * @param arr String array
	 * @return Double array of parsed string variables
	 */
	private static double[] StringtoDub(String[] arr) {
		double[] ret = new double[arr.length];
		for(int index = 0; index < arr.length; index++) {
			ret[index] = Double.parseDouble(arr[index]);
		}
		
		return ret;
	}
	
	/**
	 * Turns an ArrayList of double arrays into a 2D double array
	 * @param list ArrayList of double arrays
	 * @return Double type 2D array of the arrays found in the given arrayList
	 */
	
	private static double[][] buildArray(ArrayList<double[]> list){
		double[][] ret = new double[list.size()][list.get(0).length];
		
		for(int index = 0; index < ret.length; index++) {
			ret[index] = list.get(index);
		}
		
		return ret;
	}
	
	/**
	 * Turns an ArrayList of 2-Dimensional Arrays into a 3-Dimensional Array<br>
	 * Will be used to build Connection Layers of each Unit
	 * @param list ArrayList of 2D arrays
	 * @return 3 Dimensional array
	 */
	private static double[][][] buildCLayers(ArrayList<double[][]> list){
		double[][][] ret = new double[list.size()][list.get(0).length][list.get(0)[0].length];
		
		for(int index = 0; index < ret.length; index++) {
			ret[index] = list.get(index);
		}
		
		return ret;
	}
	
}
