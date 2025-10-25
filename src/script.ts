import { match, P } from 'ts-pattern';
import { styled } from 'styled-components';
import { Chalk } from 'chalk';

const datasetURL: string = '/MNIST-test-dataset.json.gz';
class Matrix {
	public data: number[][];

	constructor(width: number, height: number, data?: number[][]) {
		let matrix: number[][] = new Array(height);
		for (let i = 0; i < matrix.length; i++) { matrix[i] = new Array(width).fill(0) }
		this.data = (data ? data : matrix);
	}

	public get transpose(): Matrix {
		let matrixTransposed = new Matrix(this.data.length, this.data[0].length);
		for (let i = 0; i < this.data.length; i++) {
		for (let j = 0; j < this.data[i].length; j++) {
			matrixTransposed.data[j][i] = this.data[i][j];
		} }

		return matrixTransposed
	}

	public hadamardProduct(matrix: Matrix): Matrix {
		if (this.data.length !== matrix.data.length || this.data[0].length !== matrix.data[0].length) { throw new Error(`Matrix dimension mismatch hadamard ${this.data.length} must equal ${matrix.data.length} and ${this.data[0].length} must equal ${matrix.data[0].length}`) }

		let result = new Matrix(this.data[0].length, this.data.length);

		for (let i = 0; i < matrix.data.length; i++) {
		for (let j = 0; j < matrix.data[0].length; j++) {
			result.data[i][j] = this.data[i][j] * matrix.data[i][j];
		}
		}

		return result
	}

	public multiply(matrix: Matrix): Matrix {
		if (this.data[0].length !== matrix.data.length) { throw new Error(`Matrix dimension mismatch dot ${this.data[0].length} must equal ${matrix.data.length}`) }

		let result = new Matrix(matrix.data[0].length, this.data.length);

		for (let i = 0; i < this.data.length; i++) {
		for (let j = 0; j < matrix.data[0].length; j++) {
			let sum: number = 0;

			for (let k = 0; k < this.data[0].length; k++) {
			sum += this.data[i][k] * matrix.data[k][j];
			}

			result.data[i][j] = sum;
		} }

		return result
	}

	public scale(scalar: number): Matrix {
    let result = new Matrix(this.data[0].length, this.data.length, this.data);

    for (let i = 0; i < this.data.length; i++) {
    for (let j = 0; j < this.data[0].length; j++) {
			result.data[i][j] *= scalar;
    } }

    return result
	}
}

class Vector extends Matrix {
	constructor(values: number[]) {
		super(values.length, 1, [values.slice()]);
	}

	public dot(other: Vector | Matrix): number | Matrix {
		if (other instanceof Vector) {
			if (this.data[0].length !== other.data[0].length) { throw new Error('Vector dimension mismatch for dot product') }
			
			let sum: number = 0;
			for (let i = 0; i < this.data[0].length; i++) {
				sum += this.data[0][i] * other.data[0][i];
			}

			return sum
		}

		if (this.data[0].length !== other.data.length) { throw new Error('Matrix dimension mismatch for dot') }

		return this.multiply(other);
	}
}

class Network {
	public layers: Layer[];
	public activationFunction: ActivationFunction;
	constructor(activationFunction: ActivationFunction, neurons: number[]) {
		this.activationFunction = activationFunction;

		if (neurons.length < 2) { throw new Error('Network must have at least 2 layers (input and output)') }

		this.layers = [];
		for (let i: number = 0; i < neurons.length; i++) {
			let previousNeurons: number = (i == 0 ? 0 : neurons[i - 1]);
			let isInputLayer: boolean = (i == 0);
			let isOutputLayer: boolean = (i == neurons.length - 1);

			if (isInputLayer) { this.layers.push(new InputLayer(neurons[i])) }
			else if (isOutputLayer) { this.layers.push(new OutputLayer(neurons[i], previousNeurons)) }
			else { this.layers.push(new Layer(neurons[i], previousNeurons)) }
		}
	}

	public output(inputData: Vector): Vector {
		if (inputData.data[0].length !== this.layers[0].neurons.length) { throw new Error('Input size does not match input layer size') }

		for (let i = 0; i < this.layers[0].neurons.length; i++) {
			this.layers[0].neurons[i].value = inputData.data[0][i];
		}

		let currentOutput: Vector = inputData;

		for (let i = 1; i < this.layers.length; i++) {
			let weights: number[][] = [];
			for (let j = 0; j < this.layers[i].neurons.length; j++) { weights.push(this.layers[i].neurons[j].weights) }
			
			let weightMatrix = new Matrix(weights[0].length, weights.length, weights);

			let biases: number[] = [];
			for (let j = 0; j < this.layers[i].neurons.length; j++) { biases.push(this.layers[i].neurons[j].bias) }

			let preActivation = new Vector(weightMatrix.multiply(currentOutput.transpose).data.map(row => row[0]));
			for (let j = 0; j < preActivation.data[0].length; j++) { preActivation.data[0][j] += biases[j] }

			let isOutputLayer: boolean = (i == this.layers.length - 1);

			if (isOutputLayer) {
				let maxVal: number = Math.max(...preActivation.data[0]);
				let expValues: number[] = [];
				let sumExp: number = 0;
					
				for (let j = 0; j < preActivation.data[0].length; j++) {
					let expVal: number = Math.exp(preActivation.data[0][j] - maxVal);
					expValues.push(expVal);
					sumExp += expVal;
				}
					
				let softmaxOutput: number[] = [];
				for (let j = 0; j < expValues.length; j++) { softmaxOutput.push(expValues[j] / sumExp) }
				
				currentOutput = new Vector(softmaxOutput);
			} else {
				let activated: number[] = [];
				for (let j = 0; j < preActivation.data[0].length; j++) {
					activated.push(this.activationFunction(preActivation.data[0][j]));
				}
				currentOutput = new Vector(activated);
			}

			for (let j = 0; j < this.layers[i].neurons.length; j++) {
				this.layers[i].neurons[j].value = currentOutput.data[0][j];
			}
		}

		return currentOutput
	}

	public train(inputData: Vector, expectedOutput: Vector, learningRate: number): number {
		let output: Vector = this.output(inputData);

		let crossEntropyLoss: number = 0;
		for (let i = 0; i < output.data[0].length; i++) {
    	if (expectedOutput.data[0][i] > 0) {
        crossEntropyLoss -= expectedOutput.data[0][i] * Math.log(output.data[0][i] + 1e-15);
    	}
		}

		let activations: Vector[] = [];
		let deltas: Vector[] = [];

		for (let i = 0; i < this.layers.length; i++) {
			let values: number[] = [];
			for (let j = 0; j < this.layers[i].neurons.length; j++) {
				values.push(this.layers[i].neurons[j].value as number);
			}
			activations.push(new Vector(values));
		}

		let outputDelta: number[] = [];
		for (let i = 0; i < output.data[0].length; i++) {
			outputDelta.push(output.data[0][i] - expectedOutput.data[0][i]);
		}
		deltas.push(new Vector(outputDelta));

		for (let i = this.layers.length - 2; i > 0; i--) {
			let weights: number[][] = [];
			for (let j = 0; j < this.layers[i + 1].neurons.length; j++) {
				weights.push(this.layers[i + 1].neurons[j].weights);
			}
			let weightMatrix = new Matrix(weights[0].length, weights.length, weights);

			let nextDelta: Vector = deltas[deltas.length - 1];
			let nextDeltaMatrix = new Matrix(1, nextDelta.data[0].length, [nextDelta.data[0]]);
			let errorPropagated: Matrix = weightMatrix.transpose.multiply(nextDeltaMatrix.transpose);

			let currentDelta: number[] = [];
			for (let j = 0; j < this.layers[i].neurons.length; j++) {
				let activation: number = activations[i].data[0][j];
				let derivative: number = activation * (1 - activation);
				currentDelta.push(errorPropagated.data[j][0] * derivative);
			}
			deltas.push(new Vector(currentDelta));
		}

		deltas.reverse();

		for (let i = 1; i < this.layers.length; i++) {
			let delta: Vector = deltas[i - 1];
			let previousActivation: Vector = activations[i - 1];

			for (let j = 0; j < this.layers[i].neurons.length; j++) {
				let neuron: Neuron = this.layers[i].neurons[j];

				for (let k = 0; k < neuron.weights.length; k++) {
					neuron.weights[k] -= learningRate * delta.data[0][j] * previousActivation.data[0][k];
				}

				neuron.bias -= learningRate * delta.data[0][j];
			}
		}

		return crossEntropyLoss;
	}
}

class Layer {
	public neurons: Neuron[];
	constructor(neurons: number, previousNeurons: number, layerType?: LayerType) {
		this.neurons = new Array(neurons).fill(null).map(() => 
		match(layerType)
			.with(LayerType.Input, () => new InputNeuron())
			.with(LayerType.Output, () => new OutputNeuron(previousNeurons))
			.with(LayerType.Hidden, () => new Neuron(previousNeurons))
			.otherwise(() => new Neuron(previousNeurons))
		);
	}
}

class InputLayer extends Layer {
	constructor(neurons: number) {
		super(neurons, 0, LayerType.Input);
	}
}

class OutputLayer extends Layer {
	constructor(neurons: number, previousNeurons: number) {
		super(neurons, previousNeurons, LayerType.Output);
	}
}

class Neuron {
	public weights: number[];
	public bias: number;
	public value: number | null;
	public neuronType: NeuronType;
	constructor(weights: number) {
		this.weights = (new Array(weights).fill(null)).map(() => (Math.random() - 0.5) * 2 * Math.sqrt(2 / weights) );
		this.bias = Math.random();
		this.value = null;
		this.neuronType = NeuronType.Hidden;
	}
}



class InputNeuron extends Neuron {
	constructor() {
		super(0);
		this.neuronType = NeuronType.Input;
	}
}

class OutputNeuron extends Neuron {
	constructor(weights: number) {
		super(weights);
		this.neuronType = NeuronType.Output;
	}

	public compute(inputs: number[], activationFunction: ActivationFunction): number {
		if (inputs.length !== this.weights.length) { throw new Error('Input length does not match weights length') }

		let sum: number = 0;

		for (let i = 0; i < inputs.length; i++) { sum += inputs[i] * this.weights[i] }

		let output: number = activationFunction(sum + this.bias);
		this.value = output;

		return output
	}
}

async function loadDataset(url: string): Promise<DataType> {
	let response: Response = await fetch(url);
	
	if (!response.ok) { throw new Error(`HTTP ${response.status}`) }
	
	let text: string = await response.text();
	let data: DataType = JSON.parse(text);
	
	return data
}

let dataset: DataType = await loadDataset(datasetURL);
let datasetLength: number = dataset.length;

function drawData(index: number, span: HTMLSpanElement, canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): void {
	let pixelSize: number = canvas.width / 28;
	
	for (let i = 0; i < 28; i++) {
	for (let j = 0; j < 28; j++) {
		let color: number = dataset[index].image[j * 28 + i];
		ctx.fillStyle = `rgb(${color}, ${color}, ${color})`;
		ctx.fillRect(i * pixelSize, j * pixelSize, pixelSize + 1, pixelSize + 1);
	}	}
	
	span.innerText = `${dataset[index].label}`;
}

type ActivationFunction = (n: number) => number

enum LayerType {
	Input = 'INPUT_LAYER',
	Hidden = 'HIDDEN_LAYER',
	Output = 'OUTPUT_LAYER'
}

enum NeuronType {
	Input = 'INPUT_NEURON',
	Hidden = 'HIDDEN_NEURON',
	Output = 'OUTPUT_NEURON'
}

type Tuple<T, N extends number> = N extends N 
  ? number extends N 
    ? T[] 
    : _TupleOf<T, N, []> 
  : never;

type _TupleOf<T, N extends number, R extends unknown[]> = R['length'] extends N 
  ? R 
  : _TupleOf<T, N, [T, ...R]>;

type Digit = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;
type DataType = { image: Tuple<number, 784>, label: Digit }[];

let toOneHotEncodng = (digit: Digit): Tuple<number, 10> => 
	match(digit)
		.returnType<Tuple<number, 10>>()
		.with(0, () => [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		.with(1, () => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
		.with(2, () => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
		.with(3, () => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
		.with(4, () => [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
		.with(5, () => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
		.with(6, () => [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
		.with(7, () => [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
		.with(8, () => [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
		.with(9, () => [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
		.exhaustive();

//////////////////////////////////////////////////////////////////////////////////////////////////////
//                                          IMPLEMENTATION                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////

let learningRate: number = 0.01;
let neuralNetworkDimensions: [784, ...number[], 10] = [784, 128, 128, 10];
let passes: number = 5;

let canvasSize: number = 300;

let canvas1 = document.getElementById('canvas1') as HTMLCanvasElement;
canvas1.width = canvasSize;
canvas1.height = canvasSize;
canvas1.style.border = '1px solid red';

let ctx1 = canvas1.getContext('2d') as CanvasRenderingContext2D;

let span = document.querySelector('span') as HTMLSpanElement;
span.style.display = `block`;

let button = document.querySelector('button') as HTMLButtonElement;
button.innerText = 'New number';
button.style.display = `block`;

let sigmoid: ActivationFunction = (x) => 1 / (1 + Math.exp(-x));

let net = new Network(
	sigmoid,
	neuralNetworkDimensions
);

drawData(0, span, canvas1, ctx1);

button.addEventListener('click', () => {
	let randomIndex: number = Math.floor(Math.random() * datasetLength);
	drawData(randomIndex, span, canvas1, ctx1);
	
	let input = new Vector(dataset[randomIndex].image.map(pixel => pixel / 255));
	let output: Vector = net.output(input);
	let prediction: number = output.data[0].indexOf(Math.max(...output.data[0]));

	// eslint-disable-next-line no-console
	console.log(`Actual: ${dataset[randomIndex].label}, Predicted: ${prediction}`);
});


let isTraining: boolean = false;
let shouldStop: boolean = false;

async function trainNetwork(): Promise<void> {
	if (isTraining) { return }
	
	isTraining = true;
	shouldStop = false;
	
	let error: number = 0;
	
	for (let pass: number = 0; pass < passes; pass++) {
		if (shouldStop) { break }
		
		for (let i = 0; i < datasetLength; i++) {
			if (shouldStop) { break }
			
			let input = new Vector(dataset[i].image.map(pixel => pixel / 255));
			let expected = new Vector(toOneHotEncodng(dataset[i].label));
			error = net.train(input, expected, learningRate);

			// eslint-disable-next-line no-console
			if (i % 200 == 0) { console.log(`Pass ${pass + 1}, Sample ${i}: ${error}`) }
			
			if (i % 50 == 0) { await new Promise(resolve => setTimeout(resolve, 0)) }
		}
		
		// eslint-disable-next-line no-console
		if (!shouldStop) { console.log(`Pass ${pass + 1} complete. Final error: ${error}`) }
	}

	// eslint-disable-next-line no-console
	if (!shouldStop) { console.log('Training complete!'); }
	
	isTraining = false;
}

(window as any).stopTraining = () => { shouldStop = true };

trainNetwork();
