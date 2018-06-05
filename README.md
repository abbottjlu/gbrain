gbrain
============
April 2 2018
<h1>GPU Javascript Library for Machine Learning</h1>

<p>The creation of this library comes after investigation and work about how avoid out to CPU when perform GPU neural network infference/training along every neural network layers.</p>
<p>I achieved a solution to this for using an Adjacency Matrix which allow know the relation easily on GPU. It's more used in real-time Graph systems GPU based but I can't saw any library using it over GPU neural networks.</p>
<p>Exist a limitation of 4096 neurons by the fact of use an Adjacency Matrix but through this system the leak to CPU is avoided on the middle layers and the CPU infference read is only performed in the last layer (out layer). On entire WebGL applications that may need a neural network system this technique would allow entire neural network execution over GPU.</p>   
<p>At the same time batch is executed in the same way, allowing inyect at this moment until 7 direct experiences at the same time per tick for training or 7 direct infferences at the same time per tick and WebGL context.</p>
<p>The Reinforcement Learning class is practically the Karpathy's RL module (ConvNetJS) modified.</p>
<p>
<a href="http://stormcolour.appspot.com/gbrain/demos/linear-regression-RL/">- ConvNetJS Reinforcement Learning demo integration</a><br />
<a href="http://stormcolour.appspot.com/gbrain/demos/linear-regression-RL/"><img src="demos/linear-regression-RL/capture.jpg" /></a><br />
<a href="http://stormcolour.appspot.com/gbrain/demos/linear-regression-RL-convolution/">- Reinforcement Learning + convolution</a><br />
<a href="http://stormcolour.appspot.com/gbrain/demos/linear-regression-RL-convolution/"><img src="demos/linear-regression-RL-convolution/capture.png" /></a>
</p>
<h2>How it works</h2>
<a href="demos/_RESOURCES/gbrain.jpg"><img src="demos/_RESOURCES/gbrain.jpg" style="width:500px"/></a> 
<p>We not need element-wise matrixs and send information to CPU on every layer result.</p>
<p>On backpropagation the weight data is updated over the Adjacency Matrix</p>
<p>Activation function is included inside own neuron, avoiding to have propagate it to any reluctance layer and so we gain better performance.</p>
<p>At this moment only linear regression with leaky-relu activation function is implemented.</p>
<p>I have been able to learn about this algorithm especially by the Andrew NG Machine Learning course, <a href="https://github.com/karpathy/convnetjs">Karpathy's ConvNetJS</a>, <a href="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/">the Matt Mazur paper</a>, Miguel Ángel Lobato & users that shared information on internet. Thanks.</p>