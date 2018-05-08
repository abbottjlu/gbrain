import {GBrain} from "./gbrain";
import {Plot} from "./Plot.class";
import {AvgWin} from "./AvgWin.class";
import Draggabilly from 'draggabilly';

/**
 * ConvNetJS Reinforcement Learning Module (https://github.com/karpathy/convnetjs)
 */
/**
 * @class
 * @param {Object} jsonIn
 * @param {HTMLElement} jsonIn.target
 * @param {Object} [jsonIn.dimensions={width: Int, height: Int}]
 * @param {int} jsonIn.num_inputs
 * @param {int} jsonIn.num_actions
 * @param {int} jsonIn.temporal_window
 * @param {int} jsonIn.experience_size
 * @param {number} jsonIn.gamma
 * @param {number} jsonIn.epsilon_min
 * @param {number} jsonIn.epsilon_test_time
 * @param {int} jsonIn.start_learn_threshold
 * @param {int} jsonIn.batch_repeats
 * @param {int} jsonIn.learning_steps_total
 * @param {int} jsonIn.learning_steps_burnin
 * @param {Array<Object>} jsonIn.layer_defs
 */

export class GBrainRL {
    constructor(jsonIn) {
        this.gbrain = null;

        this.num_inputs = jsonIn.num_inputs;
        this.num_actions = jsonIn.num_actions;
        this.temporal_window = jsonIn.temporal_window;
        this.experience_size = jsonIn.experience_size;
        this.gamma = jsonIn.gamma;
        this.epsilon_min = jsonIn.epsilon_min;
        this.epsilon_test_time = jsonIn.epsilon_test_time;
        this.start_learn_threshold = jsonIn.start_learn_threshold;
        this.learning_steps_total = jsonIn.learning_steps_total;
        this.learning_steps_burnin = jsonIn.learning_steps_burnin;

        this.net_inputs = this.num_inputs * this.temporal_window + this.num_actions * this.temporal_window + this.num_inputs;
        this.window_size = Math.max(this.temporal_window, 2);

        this.state_window = new Array(this.window_size);
        this.action_window = new Array(this.window_size);
        this.reward_window = new Array(this.window_size);
        this.net_window = new Array(this.window_size);

        this.experience = [];


        this.onLearned = null;

        this.age = 0;
        this.ageEpoch = 0;
        this.epoch = 0;
        this.epsilon = 1.0;
        this.loss = 0.0;

        this.latest_reward = 0;
        this.last_input_array = null;
        this.forward_passes = 0;
        this.learning = true;

        this.sweep = 0;
        this.sweepMax = 200;
        this.sweepDir = 0;
        this.sweepEnable = false;
        this.plotEnable = true;

        this.arrInputs = [];
        this.arrTargets = [];

        this.lastTotalError = 0;


        this.showOutputWeighted = false;
        this.showWD = false;

        let target = document.createElement("div");
        document.getElementsByTagName("body")[0].appendChild(target);
        target.style.width = "950px";
        target.innerHTML = `
        <div style="min-width:300px; font-size:12px; box-shadow:rgba(0, 0, 0, 0.683594) 3px 3px 8px 1px,rgb(255, 255, 255) 0px 0px 5px 0px inset; border-radius:5px;">
            <div id="elGbrainWindowHandle" style="border-top-left-radius:5px; border-top-right-radius:5px; width:100%; background:rgba(200,200,200,0.7); cursor:move;	display:table;">
                <div style="padding-left:5px; font-size:14px; color:#000; font-weight:bold;	display:table-cell;	vertical-align:middle;">GBrain</div>
                <div style="width:22px;	padding:2px; display:table-cell; vertical-align:middle;">
                    <div class="SECmenuTitleCloseImg"></div>
                </div>
            </div>
            <div style="border-bottom-left-radius:5px; border-bottom-right-radius:5px; min-width:220px;	cursor:default;	padding:5px; color:#FFF; background:rgba(50,50,50,0.95); overflow-y:auto;">
                <div style="display:inline-block;width:400px;">
                    Loss
                    <canvas id="elPlotLoss" style="background:#FFF"></canvas><br />
                    Epsilon
                    <canvas id="elPlotEpsilon" style="background:#FFF"></canvas><br />
                    <button id="BTNID_PLOTMODE" style="display:inline-block;">Plot mode</button>
                    <button id="BTNID_PLOTENABLE" style="display:inline-block;">Enable plot</button>
                    <div id="el_info"></div>
                    <div>
                        View weight*neuron output<input title="weight*output" type="checkbox" id="elem_enableOutputWeighted"/><br />
                        View weight dynamics<input title="weight dynamics" type="checkbox" id="elem_enableWeightDynamics"/>
                    </div>
                    <button id="BTNID_SWEEPEPSILON" style="display:inline-block;">Sweep epsilon</button>
                    <button id="BTNID_STOP" style="display:inline-block;">Stop train</button>
                    <button id="BTNID_RESUME" style="display:inline-block;">Resume train</button>
                    <button id="BTNID_TOJSON" style="display:inline-block;">Output model in console</button>
                    <button id="BTNID_TOLSJSON" style="display:inline-block;">Save model in LocalStorage</button>
                    <button id="BTNID_FROMLSJSON" style="display:inline-block;">Load model from LocalStorage</button>
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
            (this.showOutputWeighted === false) ? this.gbrain.enableShowOutputWeighted() : this.gbrain.disableShowOutputWeighted();
            this.showOutputWeighted = !this.showOutputWeighted;
        });
        target.querySelector("#elem_enableWeightDynamics").addEventListener("click", () => {
            (this.showWD === false) ? this.gbrain.enableShowWeightDynamics() : this.gbrain.disableShowWeightDynamics();
            this.showWD = !this.showWD;
        });

        target.querySelector("#BTNID_SWEEPEPSILON").addEventListener("click", () => {
            this.sweepEnable = (this.sweepEnable !== true);
        });

        target.querySelector("#BTNID_STOP").addEventListener("click", () => {
            this.stopLearning();
        });

        target.querySelector("#BTNID_RESUME").addEventListener("click", () => {
            this.resumeLearning();
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

        let dragg = new Draggabilly( target, {
            handle: '#elGbrainWindowHandle'
        });
        target.style.left = (-target.getBoundingClientRect().left+100)+"px";
        target.style.top = (-target.getBoundingClientRect().top+100)+"px";

        this.avgLossWin = new AvgWin();

        this.plotLoss = new Plot();
        this.plotLossCanvas = target.querySelector("#elPlotLoss");

        this.plotEpsilon = new Plot();
        this.plotEpsilonCanvas = target.querySelector("#elPlotEpsilon");

        this.clock = 0;

        if(jsonIn.layer_defs !== undefined && jsonIn.layer_defs !== null) {
            this.gbrain = new GBrain({  "target": target.querySelector("#el_gbrainDisplay"),
                "dimensions": {"width": 500, "height": 500},
                "batch_repeats": jsonIn.batch_repeats,
                "learning_rate": jsonIn.learning_rate});
            this.gbrain.makeLayers(jsonIn.layer_defs);
        }
    }

    fromJson(jsonIn) {
        this.gbrain.fromJson(jsonIn);
    };

    /**
     * @returns {String}
     */
    toJson() {
        return this.gbrain.toJson();
    };

    getNetInput(xt) {
        // return s = (x,a,x,a,x,a,xt) state vector.
        // It's a concatenation of last window_size (x,a) pairs and current state x
        let w = [];
        w = w.concat(xt); // start with current state
        // and now go backwards and append states and actions from history temporal_window times
        let n = this.window_size;
        for(let k=0; k < this.temporal_window; k++) {
            // state
            w = w.concat(this.state_window[n-1-k]);
            // action, encoded as 1-of-k indicator vector. We scale it up a bit because
            // we dont want weight regularization to undervalue this information, as it only exists once
            let action1ofk = new Array(this.num_actions);
            for(let q=0; q < this.num_actions; q++)
                action1ofk[q] = 0.0;
            action1ofk[this.action_window[n-1-k]] = 1.0/* *this.num_inputs*/;
            w = w.concat(action1ofk);
        }
        return w;
    };

    random_action() {
        return Math.floor(Math.random()*this.num_actions);
    };

    policy(s, onP) {
        // compute the value of doing any action in this state
        // and return the argmax action and its value
        this.gbrain.forward(s, (maxacts) => {
            onP(maxacts);
        });
    };

    pushWindow(input_array, net_input, action) {
        // remember the state and action we took for backward pass
        this.state_window.shift();
        this.state_window.push(input_array);

        this.net_window.shift();
        this.net_window.push(net_input);

        this.action_window.shift();
        this.action_window.push(action);
    };

    stopLearning() {
        this.learning = false;
    };

    resumeLearning() {
        this.learning = true;
        this.forward_passes = 0;

        this.state_window = new Array(this.window_size);
        this.action_window = new Array(this.window_size);
        this.reward_window = new Array(this.window_size);
        this.net_window = new Array(this.window_size);
    };

    forward(input_array, onAction) {
        this.forward_passes++;
        this.last_input_array = input_array;

        let action = null;
        let net_input = this.getNetInput(input_array);
        if(this.forward_passes > this.temporal_window) { // we have enough to actually do something reasonable
            if(this.learning === true) {
                this.epsilon = Math.min(1.0, Math.max(this.epsilon_min, 1.0-(this.age - this.learning_steps_burnin)/(this.learning_steps_total - this.learning_steps_burnin)));
                if(this.sweepEnable === true) {
                    if(this.sweep >= this.sweepMax)
                        this.sweepDir = -1;
                    else if(this.sweep <= 0)
                        this.sweepDir = 1;
                    this.sweep+=this.sweepDir;
                    if(this.latest_reward > 0) {
                        let rewardMultiplier = 1.0-Math.min(1, Math.max(0.0, this.latest_reward*2));
                        let sweepMultiplier = (Math.abs(this.sweep)/this.sweepMax);
                        this.epsilon = Math.max(this.epsilon_min, rewardMultiplier*sweepMultiplier*this.epsilon);
                    }
                }
            } else
                this.epsilon = this.epsilon_test_time;

            let rf = Math.random();
            if(rf < this.epsilon) {
                // choose a random action with epsilon probability
                action = this.random_action();
                this.pushWindow(input_array, net_input, action);
                onAction(action);
            } else {
                // otherwise use our policy to make decision
                this.policy(net_input, (maxact) => {
                    this.pushWindow(input_array, net_input, maxact[0].action);
                    onAction(maxact[0].action);
                });
            }
        } else {
            // pathological case that happens first few iterations
            // before we accumulate window_size inputs
            //net_input = [];
            action = this.random_action();
            this.pushWindow(input_array, net_input, action);
            onAction(action);
        }
    };

    backward(reward, _onLearned) {
        this.onLearned = _onLearned;
        this.latest_reward = reward;

        this.clock++;
        this.el_info.innerHTML =  "epsilon: "+this.epsilon+"<br />"+
            "reward: "+this.latest_reward+"<br />"+
            "age: "+this.age+"<br />"+
            "average Q-learning loss: "+this.loss+"<br />"+
            "current learning rate: "+this.gbrain.currentLearningRate;

        if(this.learning === false || this.forward_passes === 0) {
            this.onLearned();
        } else {
            //this.average_reward_window.add(reward); TODO
            this.reward_window.shift();
            this.reward_window.push(reward);

            if(this.ageEpoch === this.experience_size) {
                this.epoch++;
                this.ageEpoch = 0;
                let dec_rate = 1.0;
                //this.gbrain.setLearningRate((1.0/(1.0+dec_rate*this.epoch))*this.gbrain.initialLearningRate);
            }
            this.age++;
            this.ageEpoch++;


            // it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
            // (given that an appropriate number of state measurements already exist, of course)
            if(this.forward_passes > (this.temporal_window+1)) {
                let n = this.window_size;
                let e = {   "state0": this.net_window[n-2],
                            "action0": this.action_window[n-2],
                            "reward0": this.reward_window[n-2],
                            "state1": this.net_window[n-1]};
                if(this.experience.length < this.experience_size)
                    this.experience.push(e);
                else {
                    let r = Math.floor(Math.random()*this.experience.length);
                    this.experience[r] = e;
                }

                if(this.experience.length > this.start_learn_threshold) {
                    let bEntries = [];
                    let state1_entries = [];
                    for(let n=0; n < this.gbrain.graph.batch_repeats*this.gbrain.graph.gpu_batch_size; n++) {
                        let e = this.experience[Math.floor(Math.random()*this.experience.length)];
                        bEntries.push(e);
                        for(let nb=0; nb < e.state1.length; nb++)
                            state1_entries.push(e.state1[nb]);
                    }
                    this.policy(state1_entries, (maxact) => {
                        this.arrInputs = [];
                        this.arrTargets = [];

                        for(let n=0; n < this.gbrain.graph.batch_repeats*this.gbrain.graph.gpu_batch_size; n++) {
                            let r = bEntries[n].reward0 + this.gamma * maxact[n].value;
                            let ystruct = {dim: bEntries[n].action0, val: r};

                            this.arrTargets.push(ystruct);

                            for(let nb=0; nb < bEntries[n].state0.length; nb++)
                                this.arrInputs.push(bEntries[n].state0[nb]);
                        }

                        this.gbrain.forward(this.arrInputs, (data) => {
                            this.gbrain.train(this.arrTargets, (loss) => {

                                this.loss = loss/(this.gbrain.graph.batch_repeats*this.gbrain.graph.gpu_batch_size);
                                this.avgLossWin.add(Math.min(10.0, this.loss));

                                this.plotLoss.add(this.clock, this.avgLossWin.get_average());
                                if(this.plotEnable === true)
                                    this.plotLoss.drawSelf(this.plotLossCanvas);

                                this.plotEpsilon.add(this.clock, this.epsilon);
                                if(this.plotEnable === true)
                                    this.plotEpsilon.drawSelf(this.plotEpsilonCanvas);

                                this.onLearned(this.loss);
                            });
                        }, false);
                    });
                } else
                    this.onLearned();
            } else
                this.onLearned();
        }
    };
}
global.GBrainRL = GBrainRL;
module.exports.GBrainRL = GBrainRL;