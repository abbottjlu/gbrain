import {GBrain} from "./gbrain";

/**
 * @author Andrej (karpathy). ConvNetJS Reinforcement Learning Module (https://github.com/karpathy/convnetjs)
 */
/**
 * @class
 */
export class GBrainRL {
    /**
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
     * @param {number} jsonIn.learning_rate
     * @param {int} jsonIn.learning_steps_total
     * @param {int} jsonIn.learning_steps_burnin
     * @param {Array<Object>} jsonIn.layer_defs
     */
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

        this.windows = [];
        for(let n=0; n < 7; n++) {
            this.windows[n] = {};
            this.windows[n].state_window = new Array(this.window_size);
            this.windows[n].action_window = new Array(this.window_size);
            this.windows[n].reward_window = new Array(this.window_size);
            this.windows[n].net_window = new Array(this.window_size);
        }

        this.experience = [];

        this.loss = 0.0;
        this.latest_reward = 0;
        this.learning = true;

        this.onLearned = null;

        this.clock = 0;

        //this.ageEpoch = 0;
        //this.epoch = 0;

        this.epsilon = 1.0;

        this.last_input_array = null;
        this.lastTotalError = 0;

        this.forward_passes = 0;

        this.sweep = 0;
        this.sweepMax = 200;
        this.sweepDir = 0;
        this.sweepEnable = false;

        this.arrInputs = [];
        this.arrTargets = [];

        if(jsonIn.layer_defs !== undefined && jsonIn.layer_defs !== null) {
            this.gbrain = new GBrain({  "batch_repeats": jsonIn.batch_repeats,
                                        "learning_rate": jsonIn.learning_rate,
                                        "onStopLearning": this.stopLearning.bind(this),
                                        "onResumeLearning": this.resumeLearning.bind(this),
                                        "rlMode": this.learning});
            this.gbrain.makeLayers(jsonIn.layer_defs);
        }
    }

    /** @private */
    getNetInput(iId, xt) {
        // return s = (x,a,x,a,x,a,xt) state vector.
        // It's a concatenation of last window_size (x,a) pairs and current state x
        let w = [];
        w = w.concat(xt); // start with current state
        // and now go backwards and append states and actions from history temporal_window times
        let n = this.window_size;
        for(let k=0; k < this.temporal_window; k++) {
            // state
            w = w.concat(this.windows[iId].state_window[n-1-k]);
            // action, encoded as 1-of-k indicator vector. We scale it up a bit because
            // we dont want weight regularization to undervalue this information, as it only exists once
            let action1ofk = new Array(this.num_actions);
            for(let q=0; q < this.num_actions; q++)
                action1ofk[q] = 0.0;
            action1ofk[this.windows[iId].action_window[n-1-k]] = 1.0/* *this.num_inputs*/;
            w = w.concat(action1ofk);
        }
        return w;
    };

    /** @private */
    random_action() {
        return Math.floor(Math.random()*this.num_actions);
    };

    /** @private */
    policy(s, onP) {
        // compute the value of doing any action in this state
        // and return the argmax action and its value
        this.gbrain.forward(s, (maxacts) => {
            onP(maxacts);
        });
    };

    /** @private */
    pushWindow(iId, input_array, net_input, action) {
        this.windows[iId].state_window.shift();
        this.windows[iId].state_window.push(input_array);

        this.windows[iId].net_window.shift();
        this.windows[iId].net_window.push(net_input);

        this.windows[iId].action_window.shift();
        this.windows[iId].action_window.push(action);
    };

    /** @private */
    stopLearning() {
        this.learning = false;
        this.forward_passes = 0;

        for(let n=0; n < 7; n++) {
            this.windows[n] = {};
            this.windows[n].state_window = new Array(this.window_size);
            this.windows[n].action_window = new Array(this.window_size);
            this.windows[n].reward_window = new Array(this.window_size);
            this.windows[n].net_window = new Array(this.window_size);
        }

        this.drawInfo();
    };

    /** @private */
    resumeLearning() {
        this.learning = true;
        this.forward_passes = 0;

        for(let n=0; n < 7; n++) {
            this.windows[n] = {};
            this.windows[n].state_window = new Array(this.window_size);
            this.windows[n].action_window = new Array(this.window_size);
            this.windows[n].reward_window = new Array(this.window_size);
            this.windows[n].net_window = new Array(this.window_size);
        }
    };

    /** @private */
    drawInfo() {
        this.gbrain.el_info.innerHTML = "learning: "+this.learning+"<br />"+
                                        "epsilon: "+this.epsilon+"<br />"+
                                        "reward: "+this.latest_reward+"<br />"+
                                        "clock: "+this.clock+"<br />"+
                                        "age: "+this.gbrain.age+"<br />"+
                                        "average Q-learning loss: "+this.loss+"<br />"+
                                        "current learning rate: "+this.gbrain.currentLearningRate;
    };

    fromJson(jsonIn) {
        this.gbrain.fromJson(jsonIn);
    };

    /**
     * @param {Array<Array<number>>} input_array
     * @param {Function} onAction
     */
    forward(input_array, onAction) {
        if(this.gbrain.learning === true) {
            this.epsilon = Math.min(1.0, Math.max(this.epsilon_min, 1.0-(this.gbrain.age - this.learning_steps_burnin)/(this.learning_steps_total - this.learning_steps_burnin)));
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

        this.forward_passes++;
        this.last_input_array = input_array;

        let entNetInput = [];
        let windowsTemp = [];
        for(let n=0; n < input_array.length; n++) {
            windowsTemp[n] = {  input_array: input_array[n],
                                net_input: null,
                                action: null};
            windowsTemp[n].net_input = this.getNetInput(n, windowsTemp[n].input_array);

            for(let nb=0; nb < windowsTemp[n].net_input.length; nb++)
                entNetInput.push(windowsTemp[n].net_input[nb]);
        }
        if(this.forward_passes > this.temporal_window) {
            let rf = Math.random();
            if(rf < this.epsilon) {
                let retActs = [];
                for(let n=0; n < input_array.length; n++) {
                    windowsTemp[n].action = this.random_action();
                    this.pushWindow(n, windowsTemp[n].input_array, windowsTemp[n].net_input, windowsTemp[n].action);
                    retActs.push(windowsTemp[n].action);
                }
                onAction(retActs);
            } else {
                this.policy(entNetInput, (maxact) => {
                    let retActs = [];
                    for(let n=0; n < input_array.length; n++) {
                        windowsTemp[n].action = maxact[n].action;
                        this.pushWindow(n, windowsTemp[n].input_array, windowsTemp[n].net_input, windowsTemp[n].action);
                        retActs.push(windowsTemp[n].action);
                    }
                    onAction(retActs);
                });
            }
        } else {
            let retActs = [];
            for(let n=0; n < input_array.length; n++) {
                windowsTemp[n].action = this.random_action();
                this.pushWindow(n, windowsTemp[n].input_array, windowsTemp[n].net_input, windowsTemp[n].action);
                retActs.push(windowsTemp[n].action);
            }
            onAction(retActs);
        }
    };

    /**
     * @param {number} reward
     * @param {Function} _onLearned
     */
    backward(reward, _onLearned) {
        this.onLearned = _onLearned;
        this.latest_reward = reward;

        this.clock++;
        this.drawInfo();

        if(this.gbrain.learning === false) {
            this.onLearned();
        } else {
            //this.average_reward_window.add(reward); TODO
            this.windows[0].reward_window.shift();
            this.windows[0].reward_window.push(reward);

            /*if(this.ageEpoch === this.experience_size) {
                this.epoch++;
                this.ageEpoch = 0;
                let dec_rate = 1.0;
                this.gbrain.setLearningRate((1.0/(1.0+dec_rate*this.epoch))*this.gbrain.initialLearningRate);
            }
            this.ageEpoch++;*/

            // it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
            // (given that an appropriate number of state measurements already exist, of course)
            if(this.forward_passes > (this.temporal_window+1)) {
                let n = this.window_size;
                let e = {   "state0": this.windows[0].net_window[n-2],
                            "action0": this.windows[0].action_window[n-2],
                            "reward0": this.windows[0].reward_window[n-2],
                            "state1": this.windows[0].net_window[n-1]};
                if(this.experience.length < this.experience_size)
                    this.experience.push(e);
                else {
                    let r = Math.floor(Math.random()*this.experience.length);
                    this.experience[r] = e;
                }

                if(this.experience.length > this.start_learn_threshold) {
                    let bEntries = [];
                    let state1_entries = [];

                    let e = this.experience[this.experience.length-1];
                    bEntries.push(e);
                    for(let nb=0; nb < e.state1.length; nb++)
                        state1_entries.push(e.state1[nb]);

                    for(let n=1; n < this.gbrain.graph.batch_repeats*this.gbrain.graph.gpu_batch_size; n++) {
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
                                this.gbrain.avgLossWin.add(Math.min(10.0, this.loss));

                                this.gbrain.plotLoss.add(this.clock, this.gbrain.avgLossWin.get_average());
                                if(this.gbrain.plotEnable === true)
                                    this.gbrain.plotLoss.drawSelf(this.gbrain.plotLossCanvas);

                                this.gbrain.plotEpsilon.add(this.clock, this.epsilon);
                                if(this.gbrain.plotEnable === true)
                                    this.gbrain.plotEpsilon.drawSelf(this.gbrain.plotEpsilonCanvas);

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