import { match, P } from 'ts-pattern';
import { styled } from 'styled-components';
import { Chalk } from 'chalk';

class Matrix {
	public data: number[][];

	constructor(width: number, height: number, data?: number[][]) {
		let matrix: number[][] = new Array(height);
		for (let i = 0; i < matrix.length; i++) { matrix[i] = new Array(width).fill(0) }
		this.data = (data ? data : matrix);
	}

	public get transpose(): Matrix {
		let matrixTransposed = new Matrix(this.data[0].length, this.data.length);
		for (let i = 0; i < this.data.length; i++) {
		for (let j = 0; j < this.data[i].length; j++) {
			matrixTransposed.data[j][i] = this.data[i][j];
		} }

		return matrixTransposed
	}

	public hadamardProduct(matrix: Matrix): Matrix {
		if (this.data.length !== matrix.data.length || this.data[0].length !== matrix.data[0].length) { throw new Error(`Matrix dimention mismatch hadamard ${this.data.length} must equal ${matrix.data.length} and ${this.data[0].length} must equal ${matrix.data[0].length}`) }

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
		let result = new Matrix(this.data.length, this.data[0].length, this.data);

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
	constructor(activationFunction: ActivationFunction, layers: number, neurons: number[], neuronTypes: NeuronType[], layerTypes: LayerType[]) {
		this.activationFunction = activationFunction;

		if (neurons.length !== layers || neuronTypes.length !== layers || layerTypes.length !== layers) { throw new Error('Neurons, neuronTypes and layerTypes arrays must match number of layers') }

		this.layers = [];
		for (let i: number = 0; i < layers; i++) {
		let previousNeurons: number = (i == 0 ? 0 : neurons[i - 1]);

		match(layerTypes[i])
			.with(LayerType.Input, () => { this.layers.push(new InputLayer(neurons[i])) })
			.with(LayerType.Output, () =>  { this.layers.push(new OutputLayer(neurons[i], previousNeurons)) })
			.with(LayerType.Hidden, () => { this.layers.push(new Layer(neurons[i], previousNeurons)) })
			.otherwise(() => { this.layers.push(new Layer(neurons[i], previousNeurons)) });
		}
	}
	
	public output(inputData: Vector): any {} // TODO
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
		this.weights = (new Array(weights).fill(null)).map(() => Math.random());
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

//////////////////////////////////////////////////////////////////////////////////////////////////////
//                                          IMPLEMENTATION                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////

const datasetURL: string = '/MNIST-test-dataset.json.gz';

let canvasSize: number = 300;

let canvas1 = document.getElementById('canvas1') as HTMLCanvasElement;
canvas1.width = canvasSize;
canvas1.height = canvasSize;
canvas1.style.border = '1px solid black';

let ctx1 = canvas1.getContext('2d') as CanvasRenderingContext2D;

let canvas2 = document.getElementById('canvas1') as HTMLCanvasElement;
canvas1.width = canvasSize;
canvas1.height = canvasSize;
canvas1.style.border = `1px solid black`;

let ctx2 = canvas1.getContext('2d') as CanvasRenderingContext2D;

let span = document.querySelector('span') as HTMLSpanElement;
span.style.display = `block`;

let button = document.querySelector('button') as HTMLButtonElement;
button.innerText = 'New number';
button.style.display = `block`;

type Tuple<T, N extends number> = N extends N 
  ? number extends N 
    ? T[] 
    : _TupleOf<T, N, []> 
  : never;

type _TupleOf<T, N extends number, R extends unknown[]> = R['length'] extends N 
  ? R 
  : _TupleOf<T, N, [T, ...R]>;

type DataType = { image: Tuple<number, 784>, label: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 }[];

async function loadDataset(url: string): Promise<DataType> {
	let response: Response = await fetch(url);
	
	if (!response.ok) { throw new Error(`HTTP ${response.status}`) }
	
	let text: string = await response.text();
	let data: DataType = JSON.parse(text);
	
	return data;
}

let dataset: DataType = await loadDataset(datasetURL);
let datasetLength: number = dataset.length;

function drawData(index: number, span: HTMLSpanElement, canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): undefined {
	let pixelSize: number = canvas.width / 28;
	
	for (let i = 0; i < 28; i++) {
		for (let j = 0; j < 28; j++) {
			let color: number = dataset[index].image[j * 28 + i];
			ctx.fillStyle = `rgb(${color}, ${color}, ${color})`;
			ctx.fillRect(i * pixelSize, j * pixelSize, pixelSize + 1, pixelSize + 1);
		}
	}
	
	span.innerText = `${dataset[index].label}`;
}

drawData(0, span, canvas1, ctx1);

button.addEventListener('click', () => {
	drawData(Math.floor(Math.random() * (datasetLength + 1)), span, canvas1, ctx1);
})
