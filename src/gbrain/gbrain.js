import "scejs";
import {Graph} from "./Graph.class";
/**
 * @class
 * @param {Object} jsonIn
 * @param {HTMLElement} jsonIn.target
 * @param {Object} [jsonIn.dimensions={width: Int, height: Int}]
 * @param {int} jsonIn.batch_repeats
 */
export class GBrain {
    constructor(jsonIn) {
        this.project = null;
        this.graph = null;

        this.ini(jsonIn);
    }

    /**
     * @param {Object} jsonIn
     * @param {HTMLElement} jsonIn.target
     * @param {Object} [jsonIn.dimensions={width: Int, height: Int}]
     * @param {int} jsonIn.batch_repeats
     * @param {number} jsonIn.learning_rate
     * @param {WebGLRenderingContext} [jsonIn.gl=undefined]
     */
    ini(jsonIn) {
        this.sce = new SCE();
        this.sce.initialize(jsonIn);

        this.project = new Project();
        this.sce.loadProject(this.project);

        let stage = new Stage();
        this.project.addStage(stage);
        this.project.setActiveStage(stage);

        // CAMERA
        let simpleCamera = new SimpleCamera(this.sce);
        simpleCamera.setView(Constants.VIEW_TYPES.TOP);
        simpleCamera.setVelocity(1.0);
        this.sce.setDimensions(jsonIn.dimensions.width, jsonIn.dimensions.height);

        // GRID
        //let grid = new Grid(this.sce);
        //grid.generate(100.0, 1.0);

        this.graph = new Graph(this.sce,{"enableFonts":true});
        this.graph.enableNeuronalNetwork();
        this.graph.layerCount = 0;
        this.outputCount = 0;
        this.layers = [];
        this.graph.batch_repeats = jsonIn.batch_repeats;
        this.initialLearningRate = jsonIn.learning_rate;
        this.currentLearningRate = jsonIn.learning_rate;

        let mesh_point = new Mesh().loadPoint();
        //this.graph.setNodeMesh(mesh_point);
    };

    /**
     * @param {Array<Object>} layer_defs
     */
    makeLayers(layer_defs) {
        let cnl = (num, weights, isInput) => {
            this.graph.layer_defs[this.graph.layerCount].hasBias = (this.graph.layer_defs[this.graph.layerCount+1].activation === "relu" ||
                                                                    this.graph.layer_defs[this.graph.layerCount+1].type === "regression") ? 1.0 : 0.0;

            this.layers.push(this.graph.createNeuronLayer(1, num, [this.graph.layerCount*30, 0.0, 0.0, 1.0], 5.0, this.graph.layer_defs[this.graph.layerCount].hasBias, isInput));

            if(isInput === 0)
                this.graph.connectNeuronLayerWithNeuronLayer({  "neuronLayerOrigin": this.layers[this.layers.length-2],
                                                                "neuronLayerTarget": this.layers[this.layers.length-1],
                                                                "weights": ((weights !== undefined && weights !== null) ? weights : null),
                                                                "layer_neurons_count": this.layers[this.layers.length-1].length,
                                                                "layerNum": this.graph.layerCount-1,
                                                                "hasBias": this.graph.layer_defs[this.graph.layerCount].hasBias}); // TODO l.activation
        };

        this.graph.layer_defs = layer_defs;

        let lType = {   "input": (l) => {
                            cnl(l.depth, l.weights, 1);

                            this.graph.layerCount++;
                        },
                        "fc": (l) => {
                            cnl(l.num_neurons, l.weights, 0);

                            this.graph.layerCount++;
                        },
                        "regression": (l) => {
                            let offsetZ = -30.0*(l.num_neurons/2);
                            let we = l.weights;
                            let newWe = [];
                            if(l.weights !== undefined && l.weights !== null) {
                                for(let n=0; n < l.num_neurons; n++) {
                                    for(let nb=0; nb < l.weights.length; nb=nb+l.num_neurons)
                                        newWe.push(l.weights[nb+n]);
                                }
                            }
                            for(let n=0; n < l.num_neurons; n++) {
                                this.graph.addEfferentNeuron("O"+this.outputCount, [this.graph.layerCount*30, 0.0, offsetZ, 1.0]); // efferent neuron (output)
                                this.graph.connectNeuronLayerWithNeuron({   "neuronLayer": this.layers[this.layers.length-1],
                                                                            "neuron": "O"+this.outputCount,
                                                                            "weight": ((l.weights !== undefined && l.weights !== null) ? newWe.slice(0, this.layers[this.layers.length-1].length) : null),
                                                                            "layer_neurons_count": this.layers[this.layers.length-2].length*this.layers[this.layers.length-1].length,
                                                                            "layerNum": this.graph.layerCount-1});
                                if(l.weights !== undefined && l.weights !== null)
                                    newWe = newWe.slice(this.layers[this.layers.length-1].length);

                                this.outputCount++;
                                offsetZ += 30.0;
                            }
                            this.graph.layerCount++;
                        }};
        for(let n=0; n < layer_defs.length; n++) {
            let l = layer_defs[n];
            lType[l.type](l);
        }

        this.graph.createWebGLBuffers();
        this.graph.enableForceLayout();

        this.graph.setLearningRate(this.currentLearningRate);
    };

    fromJson(jsonIn) {
        let layer_defs = [];
        for(let n=0; n < jsonIn.layers.length; n++) {
            if(jsonIn.layers[n].layer_type === "input") {


            } else if(jsonIn.layers[n].layer_type === "fc") {
                jsonIn.layers[n].weights = [];
                for(let key in jsonIn.layers[n].filters[0].w) {
                    for(let nb=0; nb < jsonIn.layers[n].filters.length; nb++) {
                        jsonIn.layers[n].weights.push(jsonIn.layers[n].filters[nb].w[key]);
                    }
                }
                for(let key in jsonIn.layers[n].biases.w)
                    jsonIn.layers[n].weights.push(jsonIn.layers[n].biases.w[key]);
            } else if(jsonIn.layers[n].layer_type === "regression") {
                jsonIn.layers[n].weights = [];
                for(let key in jsonIn.layers[n].filters[0].w) {
                    for(let nb=0; nb < jsonIn.layers[n].filters.length; nb++) {
                        jsonIn.layers[n].weights.push(jsonIn.layers[n].filters[nb].w[key]);
                    }
                }
                for(let key in jsonIn.layers[n].biases.w)
                    jsonIn.layers[n].weights.push(jsonIn.layers[n].biases.w[key]);
            }
        }
        for(let n=0; n < jsonIn.layers.length; n++) {
            if(jsonIn.layers[n].layer_type === "input")
                layer_defs.push({"type": "input", "depth": jsonIn.layers[n].out_depth});
            else if(jsonIn.layers[n].layer_type === "fc")
                layer_defs.push({"type": "fc", "num_neurons": jsonIn.layers[n].out_depth, "activation": "relu", "weights": jsonIn.layers[n].weights});
            else if(jsonIn.layers[n].layer_type === "regression")
                layer_defs.push({"type": "regression", "num_neurons": jsonIn.layers[n].out_depth, "weights": jsonIn.layers[n].weights});
        }

        this.sce.target.innerHTML = "";
        this.ini({  "target": this.sce.target,
                    "dimensions": this.sce.dimensions,
                    "batch_repeats": this.graph.batch_repeats,
                    "learning_rate": this.currentLearningRate});
        this.makeLayers(layer_defs);
    };

    /**
     * @returns {String}
     */
    toJson() {
        return this.graph.toJson();
    };

    /**
     * @param {Array<number>} state
     * @param {Function} onAction
     * @param {boolean} [readOutput]
     */
    forward(state, onAction, readOutput) {
        this.graph.forward({"state": state,
                            "readOutput": readOutput,
                            "onAction": (maxacts) => {
                                onAction(maxacts);
                            }});
    };

    /**
     * @param {Array<Object>} reward [{dim: actionId for the reward , val: number}]
     * @param {Function} onTrain
     */
    train(reward, onTrain) {
        this.graph.train({  "reward": reward,
                            "onTrained": (loss) => {
                                onTrain(loss);
                            }});
    };

    setLearningRate(v) {
        this.currentLearningRate = v;
        this.graph.setLearningRate(v);
    };

    enableShowOutputWeighted() {
        this.graph.enableShowOutputWeighted();
    };

    disableShowOutputWeighted() {
        this.graph.disableShowOutputWeighted();
    };

    enableShowWeightDynamics() {
        this.graph.enableShowWeightDynamics();
    };

    disableShowWeightDynamics() {
        this.graph.disableShowWeightDynamics();
    };

    enableShowValues() {
        this.graph.enableShowValues();
    };

    disableShowValues() {
        this.graph.disableShowValues();
    };

    tick() {
        this.project.getActiveStage().tick();
    };
}
global.GBrain = GBrain;
module.exports.GBrain = GBrain;