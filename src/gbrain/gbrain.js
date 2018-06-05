import "scejs";
import {Graph} from "./Graph.class";
import {Plot} from "./Plot.class";
import {AvgWin} from "./AvgWin.class";
import Draggabilly from 'draggabilly';

/**
 * @class
 */
export class GBrain {
    constructor(jsonIn) {
        this.project = null;
        this.graph = null;

        this.ini(jsonIn);
    }

    /**
     * @param {Object} jsonIn
     * @param {int} jsonIn.batch_repeats
     * @param {number} jsonIn.learning_rate
     * @param {WebGLRenderingContext} [jsonIn.gl=undefined]
     * @param {Array<Object>} jsonIn.layer_defs
     * @param {Function} jsonIn.onStopLearning
     * @param {Function} jsonIn.onResumeLearning
     * @param {boolean} jsonIn.rlMode
     */
    ini(jsonIn) {
        this.rlMode = (jsonIn.rlMode !== undefined && jsonIn.rlMode !== null && jsonIn.rlMode === true);

        this.learning = true;
        this.age = 0;

        this.plotEnable = true;

        this.showOutputWeighted = false;
        this.showWD = false;
        this.showValues = true;

        this.windowSavedPosLeft = 0;
        this.windowSavedPosTop = 0;
        this.windowEnabled = true;

        ////////////////////////////////////////////
        // UI PANEL
        ////////////////////////////////////////////
        let epsilonStr = (rlMode) => {
            return (rlMode === true)
                ? `
                    Epsilon
                    <canvas id="elPlotEpsilon" style="background:#FFF"></canvas><br />`
                : '';
        };
        let rlStr = (rlMode) => {
            return (rlMode === true)
                        ? `
                    <button id="BTNID_SWEEPEPSILON" style="width:120px;display:inline-block;">Sweep*reward epsilon</button>
                    <button id="BTNID_STOP" style="width:120px;display:inline-block;">Stop train</button>
                    <button id="BTNID_RESUME" style="width:120px;display:inline-block;">Resume train</button>`
                        : '';
        };

        let target = null;
        let exists = false;
        if(document.getElementById("elGBrainPanel") !== null) {
            target = document.getElementById("elGBrainPanel");
            exists = true;
        } else {
            target = document.createElement("div");
            document.getElementsByTagName("body")[0].appendChild(target);
        }
        target.id = "elGBrainPanel";
        target.style.width = "950px";
        target.innerHTML = `
        <div style="font-size:12px; box-shadow:rgba(0, 0, 0, 0.683594) 3px 3px 8px 1px,rgb(255, 255, 255) 0 0 5px 0 inset; border-radius:5px;">
            <div id="elGbrainWindowHandle" style="border-top-left-radius:5px; border-top-right-radius:5px; width:100%; background:rgba(200,200,200,0.7); cursor:move;	display:table;">
                <div style="padding-left:5px;color:#000;font-weight:bold;display:table-cell;vertical-align:middle;">GBrain</div>
                <div style="width:22px;	padding:2px; display:table-cell; vertical-align:middle;">
                    <div id="elGbrainMinMax" style="font-weight:bold;cursor:pointer;">&#95;</div>
                </div>
            </div>
            <div id="elGbrainContent" style="border-bottom-left-radius:5px; border-bottom-right-radius:5px; min-width:220px;	cursor:default;	padding:5px; color:#FFF; background:rgba(50,50,50,0.95); overflow-y:auto;">
                <div style="display:inline-block;width:400px;vertical-align:top;">
                    Loss
                    <canvas id="elPlotLoss" style="background:#FFF"></canvas><br />
                    ${epsilonStr(this.rlMode)}
                    <button id="BTNID_PLOTMODE" style="display:inline-block;">Plot mode</button>
                    <button id="BTNID_PLOTENABLE" style="display:inline-block;">Enable plot</button>
                    <div id="el_info"></div>
                    <div>
                        Show weight*neuron output<input title="weight*output" type="checkbox" id="elem_enableOutputWeighted"/><br />
                        Show weight dynamics<input title="weight dynamics" type="checkbox" id="elem_enableWeightDynamics"/><br />
                        Show output values<input title="input values" type="checkbox" checked="checked" id="elem_enableShowValues"/>
                    </div>
                    ${rlStr(this.rlMode)}
                    <button id="BTNID_TOJSON" style="width:120px;display:inline-block;">Output model in console</button>
                    <button id="BTNID_TOLSJSON" style="width:120px;display:inline-block;">Save model in LocalStorage</button>
                    <button id="BTNID_FROMLSJSON" style="width:120px;display:inline-block;">Load model from LocalStorage</button>
                </div>
                <div style="display:inline-block;">
                    <div id="el_gbrainDisplay"></div>
                </div>
            </div>
        </div>
        `;
        this.el_info = target.querySelector("#el_info");

        target.querySelector("#BTNID_PLOTMODE").addEventListener("click", () => {
            this.plotLoss.currentMode = (this.plotLoss.currentMode === 0) ? 1 : 0;
            this.plotEpsilon.currentMode = (this.plotEpsilon.currentMode === 0) ? 1 : 0;
        });
        target.querySelector("#BTNID_PLOTENABLE").addEventListener("click", () => {
            this.plotEnable = (this.plotEnable !== true);
        });
        target.querySelector("#elem_enableOutputWeighted").addEventListener("click", () => {
            (this.showOutputWeighted === false) ? this.graph.enableShowOutputWeighted() : this.graph.disableShowOutputWeighted();
            this.showOutputWeighted = !this.showOutputWeighted;
        });
        target.querySelector("#elem_enableWeightDynamics").addEventListener("click", () => {
            (this.showWD === false) ? this.graph.enableShowWeightDynamics() : this.graph.disableShowWeightDynamics();
            this.showWD = !this.showWD;
        });
        target.querySelector("#elem_enableShowValues").addEventListener("click", () => {
            (this.showValues === false) ? this.graph.enableShowValues() : this.graph.disableShowValues();
            this.showValues = !this.showValues;
        });

        target.querySelector("#BTNID_TOJSON").addEventListener("click", () => {
            this.toJson();
        });

        target.querySelector("#BTNID_TOLSJSON").addEventListener("click", () => {
            localStorage.trainedModel = this.toJson();
        });
        target.querySelector("#BTNID_FROMLSJSON").addEventListener("click", () => {
            this.fromJson(JSON.parse(localStorage.trainedModel));
        });

        target.querySelector("#elGbrainMinMax").addEventListener("click", () => {
            if(this.windowEnabled === true) {
                this.windowSavedPosLeft = target.style.left;
                this.windowSavedPosTop = target.style.top;
                target.style.position = "absolute";
                target.style.left = "50";
                target.style.top = "0";
                target.style.width = "150px";
                target.querySelector("#elGbrainContent").style.display = "none";
                target.querySelector("#elGbrainMinMax").innerHTML = "&square;";
                this.windowEnabled = false;
            } else {
                target.style.position = "relative";
                target.style.left = this.windowSavedPosLeft;
                target.style.top = this.windowSavedPosTop;
                target.style.width = "950px";
                target.querySelector("#elGbrainContent").style.display = "block";
                target.querySelector("#elGbrainMinMax").innerHTML = "&#95;";
                this.windowEnabled = true;
            }
        });

        if(this.rlMode === true) {
            target.querySelector("#BTNID_SWEEPEPSILON").addEventListener("click", () => {
                this.sweepEnable = (this.sweepEnable !== true);
            });

            target.querySelector("#BTNID_STOP").addEventListener("click", () => {
                this.stopLearning();
            });

            target.querySelector("#BTNID_RESUME").addEventListener("click", () => {
                this.resumeLearning();
            });
        }

        let dragg = new Draggabilly( target, {
            handle: '#elGbrainWindowHandle'
        });
        if(exists === false) {
            target.style.left = (-target.getBoundingClientRect().left+100)+"px";
            target.style.top = (-target.getBoundingClientRect().top+100)+"px";
        }

        this.avgLossWin = new AvgWin();

        this.plotLoss = new Plot();
        this.plotLossCanvas = target.querySelector("#elPlotLoss");

        this.plotEpsilon = new Plot();
        this.plotEpsilonCanvas = target.querySelector("#elPlotEpsilon");

        ////////////////////////////////////////////
        // SCEJS
        ////////////////////////////////////////////
        jsonIn.target = target.querySelector("#el_gbrainDisplay");
        jsonIn.dimensions = {"width": 500, "height": 500};
        jsonIn.enableUI = true;

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
        this.graph.batch_repeats = (jsonIn.batch_repeats !== undefined && jsonIn.batch_repeats !== null) ? jsonIn.batch_repeats : 1;
        this.initialLearningRate = jsonIn.learning_rate;
        this.currentLearningRate = jsonIn.learning_rate;
        this.onStopLearning = jsonIn.onStopLearning;
        this.onResumeLearning = jsonIn.onResumeLearning;

        this.outputCount = 0;
        this.layerNodes = [];

        let mesh_point = new Mesh().loadPoint();
        //this.graph.setNodeMesh(mesh_point);

        if(jsonIn.layer_defs !== undefined && jsonIn.layer_defs !== null)
            this.makeLayers(jsonIn.layer_defs);
    };

    stopLearning() {
        this.learning = false;
        this.onStopLearning();
    };

    resumeLearning() {
        this.learning = true;
        this.onResumeLearning();
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
                                let newNeurons = mLayer({   "w": this.graph.layer_defs[this.graph.layerCount].out_sx,
                                                            "h": this.graph.layer_defs[this.graph.layerCount].out_sy,
                                                            "isInput": 0,
                                                            "type": l.type,
                                                            "posZ": displ[n],
                                                            "bias": ((n === displ.length-1) ? 1 : 0)});

                                mRelations({"w": this.graph.layer_defs[this.graph.layerCount].out_sx,
                                            "originLayer": this.graph.layer_defs[this.graph.layerCount-1].neurons,
                                            "targetLayer": newNeurons,
                                            "weights": ((l.weights !== undefined && l.weights !== null) ? l.weights.slice(0, wh*9) : null),
                                            "type": l.type,
                                            "layerNum": this.graph.layerCount-1,
                                            "hasBias": this.graph.layer_defs[this.graph.layerCount].hasBias,
                                            "makeBias": 0,
                                            "layer_neurons_count": this.graph.layer_defs[this.graph.layerCount-1].length,
                                            "convMatrixId": n});

                                this.graph.layer_defs[this.graph.layerCount].neurons = this.graph.layer_defs[this.graph.layerCount].neurons.concat(newNeurons);

                                if(l.weights !== undefined && l.weights !== null)
                                    l.weights = l.weights.slice(wh*9);

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

    /**
     * @param {Object} jsonIn
     */
    fromJson(jsonIn) {
        let layer_defs = [];
        for(let n=0; n < jsonIn.layers.length; n++) {
            if(jsonIn.layers[n].layer_type === "input") {


            } else if(  jsonIn.layers[n].layer_type === "fc" ||
                        jsonIn.layers[n].layer_type === "regression") {
                jsonIn.layers[n].weights = [];
                for(let key in jsonIn.layers[n].filters[0].w) {
                    for(let nb=0; nb < jsonIn.layers[n].filters.length; nb++) {
                        jsonIn.layers[n].weights.push(jsonIn.layers[n].filters[nb].w[key]);
                    }
                }
                for(let key in jsonIn.layers[n].biases.w)
                    jsonIn.layers[n].weights.push(jsonIn.layers[n].biases.w[key]);
            } else if(  jsonIn.layers[n].layer_type === "conv") {
                jsonIn.layers[n].weights = [];
                for(let nb=0; nb < jsonIn.layers[n].filters.length; nb++) {
                    for(let key in jsonIn.layers[n].filters[nb].w) {
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
        let isLearning = this.learning;
        this.ini({  "target": this.sce.target,
                    "dimensions": this.sce.dimensions,
                    "batch_repeats": this.graph.batch_repeats,
                    "learning_rate": this.currentLearningRate,
                    "onStopLearning": this.onStopLearning,
                    "onResumeLearning": this.onResumeLearning,
                    "rlMode": this.rlMode,
                    "layer_defs": layer_defs
                    });

        if(this.rlMode === true && isLearning === false)
            this.stopLearning();
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
    backward(reward, onTrain) {
        this.age++;
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