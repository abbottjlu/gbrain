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
        this.layerNodes = [];
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
        let mLayer = (jsonIn) => {
            if(jsonIn.bias !== undefined && jsonIn.bias !== null && jsonIn.bias === 1) {
                this.graph.layer_defs[this.graph.layerCount].hasBias = (this.graph.layer_defs[this.graph.layerCount+1].activation === "relu" ||
                                                                        this.graph.layer_defs[this.graph.layerCount+1].type === "regression" ||
                                                                        this.graph.layer_defs[this.graph.layerCount+1].type === "classification") ? 1.0 : 0.0;
            } else
                this.graph.layer_defs[this.graph.layerCount].hasBias = 0.0;

            return this.graph.createNeuronLayer(jsonIn.w, jsonIn.h, [this.offsetX, 0.0, jsonIn.posZ, 1.0], 5.0, this.graph.layer_defs[this.graph.layerCount].hasBias, jsonIn.isInput);
        };

        let mRelations = (jsonIn) => {
            if(jsonIn.type === "fc") {
                this.graph.connectNeuronLayerWithNeuronLayer({  "neuronLayerOrigin": jsonIn.originLayer,
                                                                "neuronLayerTarget": jsonIn.targetLayer,
                                                                "weights": ((jsonIn.weights !== undefined && jsonIn.weights !== null) ? jsonIn.weights : null),
                                                                "layer_neurons_count": jsonIn.layer_neurons_count,
                                                                "layerNum": jsonIn.layerNum,
                                                                "hasBias": jsonIn.hasBias,
                                                                "makeBias": jsonIn.makeBias}); // TODO l.activation
            } else if(jsonIn.type === "conv") {
                this.graph.connectConvXYNeuronsFromXYNeurons({  "w": jsonIn.w,
                                                                "neuronLayerOrigin": jsonIn.originLayer,
                                                                "neuronLayerTarget": jsonIn.targetLayer,
                                                                "weights": ((jsonIn.weights !== undefined && jsonIn.weights !== null) ? jsonIn.weights : null),
                                                                "layer_neurons_count": jsonIn.layer_neurons_count,
                                                                "layerNum": jsonIn.layerNum,
                                                                "hasBias": jsonIn.hasBias,
                                                                "makeBias": jsonIn.makeBias,
                                                                "convMatrixId": jsonIn.convMatrixId});
            }
        };

        let ll = (h, originLayer, weights) => {
            let offsetZ = -5.0*(h/2);
            let we = weights;
            let newWe = [];
            if(weights !== undefined && weights !== null) {
                for(let n=0; n < h; n++) {
                    for(let nb=0; nb < weights.length; nb=nb+h)
                        newWe.push(weights[nb+n]);
                }
            }
            let arr = [];
            for(let n=0; n < h; n++) {
                let name = "O "+this.outputCount;
                arr.push(name);
                this.graph.addEfferentNeuron(name, [this.offsetX, 0.0, offsetZ, 1.0]); // efferent neuron (output)
                this.graph.connectNeuronLayerWithNeuron({   "neuronLayer": originLayer,
                                                            "neuron": "O "+this.outputCount,
                                                            "weight": ((weights !== undefined && weights !== null) ? newWe.slice(0, originLayer.length) : null),
                                                            "layer_neurons_count": originLayer.length,
                                                            "layerNum": this.graph.layerCount-1});
                if(weights !== undefined && weights !== null)
                    newWe = newWe.slice(originLayer.length);

                this.outputCount++;
                offsetZ += 5.0;
            }

            return arr;
        };


        this.graph.layer_defs = layer_defs;
        this.offsetX = 0;

        let lType = {   "input": (l) => {
                            this.graph.layer_defs[this.graph.layerCount].neurons = (l.out_sx !== undefined)
                                                                                        ? mLayer({  "w": l.out_sx,
                                                                                                    "h": l.out_sy,
                                                                                                    "isInput": 1,
                                                                                                    "type": l.type,
                                                                                                    "posZ": 0,
                                                                                                    "bias": 1})
                                                                                        : mLayer({  "w": 1,
                                                                                                    "h": l.depth,
                                                                                                    "isInput": 1,
                                                                                                    "type": l.type,
                                                                                                    "posZ": 0,
                                                                                                    "bias": 1});

                            this.offsetX += ((l.out_sx !== undefined) ? 100 : 30);
                        },
                        "fc": (l) => {
                            this.graph.layer_defs[this.graph.layerCount].neurons = mLayer({ "w": 1,
                                                                                            "h": l.num_neurons,
                                                                                            "isInput": 0,
                                                                                            "type": l.type,
                                                                                            "posZ": 0,
                                                                                            "bias": 1});

                            mRelations({"w": null,
                                        "originLayer": this.graph.layer_defs[this.graph.layerCount-1].neurons,
                                        "targetLayer": this.graph.layer_defs[this.graph.layerCount].neurons,
                                        "weights": l.weights,
                                        "type": l.type,
                                        "layerNum": this.graph.layerCount-1,
                                        "hasBias": this.graph.layer_defs[this.graph.layerCount].hasBias,
                                        "makeBias": 0,
                                        "layer_neurons_count": this.graph.layer_defs[this.graph.layerCount-1].length});
                            mRelations({"w": null,
                                        "originLayer": this.graph.layer_defs[this.graph.layerCount-1].neurons,
                                        "targetLayer": this.graph.layer_defs[this.graph.layerCount].neurons,
                                        "weights": l.weights,
                                        "type": l.type,
                                        "layerNum": this.graph.layerCount-1,
                                        "hasBias": this.graph.layer_defs[this.graph.layerCount].hasBias,
                                        "makeBias": 1,
                                        "layer_neurons_count": this.graph.layer_defs[this.graph.layerCount-1].length});

                            this.offsetX += 30;
                        },
                        "conv": (l) => {
                            let displ = [-180,-120,-60,60,120,180];

                            if(l.type === "conv") {
                                this.graph.layer_defs[this.graph.layerCount].out_sx = this.graph.layer_defs[this.graph.layerCount-1].out_sx-2;
                                this.graph.layer_defs[this.graph.layerCount].out_sy = this.graph.layer_defs[this.graph.layerCount-1].out_sy-2;
                            }
                            let wh = this.graph.layer_defs[this.graph.layerCount].out_sx*this.graph.layer_defs[this.graph.layerCount].out_sy;

                            this.graph.layer_defs[this.graph.layerCount].neurons = [];
                            for(let n=0; n < displ.length; n++) {
                                let bias = (n === displ.length-1) ? 1 : 0;
                                let newNeurons = mLayer({   "w": this.graph.layer_defs[this.graph.layerCount].out_sx,
                                                            "h": this.graph.layer_defs[this.graph.layerCount].out_sy,
                                                            "isInput": 0,
                                                            "type": l.type,
                                                            "posZ": displ[n],
                                                            "bias": bias});

                                let wN = (l.weights !== undefined && l.weights !== null) ? l.weights.slice(0, wh*9) : null;

                                mRelations({"w": this.graph.layer_defs[this.graph.layerCount].out_sx,
                                            "originLayer": this.graph.layer_defs[this.graph.layerCount-1].neurons,
                                            "targetLayer": newNeurons,
                                            "weights": wN,
                                            "type": l.type,
                                            "layerNum": this.graph.layerCount-1,
                                            "hasBias": this.graph.layer_defs[this.graph.layerCount].hasBias,
                                            "makeBias": 0,
                                            "layer_neurons_count": this.graph.layer_defs[this.graph.layerCount-1].length,
                                            "convMatrixId": n});
                                this.graph.layer_defs[this.graph.layerCount].neurons = this.graph.layer_defs[this.graph.layerCount].neurons.concat(newNeurons);

                                if(this.graph.layer_defs[this.graph.layerCount].hasBias ===1 && n === displ.length-1)
                                    mRelations({"w": this.graph.layer_defs[this.graph.layerCount].out_sx,
                                                "originLayer": this.graph.layer_defs[this.graph.layerCount-1].neurons,
                                                "targetLayer": this.graph.layer_defs[this.graph.layerCount].neurons,
                                                "weights": l.weights,
                                                "type": l.type,
                                                "layerNum": this.graph.layerCount-1,
                                                "hasBias": this.graph.layer_defs[this.graph.layerCount].hasBias,
                                                "makeBias": 1,
                                                "layer_neurons_count": this.graph.layer_defs[this.graph.layerCount-1].length,
                                                "convMatrixId": n});

                                if(l.weights !== undefined && l.weights !== null)
                                    l.weights = l.weights.slice(wh*9);
                            }

                            this.offsetX += 100;
                        },
                        "regression": (l) => {
                            this.graph.layer_defs[this.graph.layerCount].neurons = ll(l.num_neurons, this.graph.layer_defs[this.graph.layerCount-1].neurons, l.weights);
                        },
                        "classification": (l) => {
                            this.graph.layer_defs[this.graph.layerCount].neurons = ll(l.num_classes, this.graph.layer_defs[this.graph.layerCount-1].neurons, l.weights);
                        }};
        for(let n=0; n < this.graph.layer_defs.length; n++) {
            let l = this.graph.layer_defs[n];
            lType[l.type](l);

            this.graph.layerCount++;
        }

        this.graph.createWebGLBuffers();
        this.graph.enableForceLayout();

        this.graph.setLearningRate(this.currentLearningRate);
    };

    fromJson(jsonIn) {
        let layer_defs = [];
        for(let n=0; n < jsonIn.layers.length; n++) {
            if(jsonIn.layers[n].layer_type === "input") {


            } else if(  jsonIn.layers[n].layer_type === "fc" ||
                        jsonIn.layers[n].layer_type === "conv" ||
                        jsonIn.layers[n].layer_type === "regression") {
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
            if(jsonIn.layers[n].layer_type === "input") {
                if(jsonIn.layers[n].out_sx !== undefined)
                    layer_defs.push({"type": jsonIn.layers[n].layer_type, "out_sx": jsonIn.layers[n].out_sx, "out_sy": jsonIn.layers[n].out_sy});
                else
                    layer_defs.push({"type": jsonIn.layers[n].layer_type, "depth": jsonIn.layers[n].out_depth});
            } else if(jsonIn.layers[n].layer_type === "fc")
                layer_defs.push({"type": jsonIn.layers[n].layer_type, "num_neurons": jsonIn.layers[n].out_depth, "activation": "relu", "weights": jsonIn.layers[n].weights});
            else if(jsonIn.layers[n].layer_type === "conv")
                layer_defs.push({"type": jsonIn.layers[n].layer_type, "num_neurons": jsonIn.layers[n].out_depth, "activation": "relu", "weights": jsonIn.layers[n].weights});
            else if(jsonIn.layers[n].layer_type === "regression")
                layer_defs.push({"type": jsonIn.layers[n].layer_type, "num_neurons": jsonIn.layers[n].out_depth, "weights": jsonIn.layers[n].weights});
            else if(jsonIn.layers[n].layer_type === "classification")
                layer_defs.push({"type": jsonIn.layers[n].layer_type, "num_classes": jsonIn.layers[n].out_depth, "weights": jsonIn.layers[n].weights});
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