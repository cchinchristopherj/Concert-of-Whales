// Web Audio API variables
let audioCtx = new (window.AudioContext || window.webkitAudioContext)();
let panner = audioCtx.createStereoPanner();
let myArrayBuffer;
let mul = 2;
let x = 0;
let y = 0;
let z = 0;
let x_update = 1;
let y_update = 0;
let z_update = 0;
let x_norm = 0;
let y_norm = 0;
let z_norm = 0;
let i = 0;
let factorI = 1;
let isPlaying = 0;

// Variable to indicate whether trained neural network is finished loading
let loading = false;

// Variable for spectrogram image
let spectrogram;

// Variables for text
let text_main = 'North Atlantic Right Whale ';
let text_ind;
let facts = ['Critically Endangered: Only ~450 are alive today',
             'Vulnerable to entanglement in fishing lines and gear',
             'Ocean noise interrupts communication and behavior',
             'Habitats dangerously close to shipping lanes and ports',
             'Threatened by vessel strikes in the Atlantic Ocean',
             'Ear wax can be used to determine age after death',
             'Migrate every fall from Canadian to Southern U.S. waters',
             'Eat by swimming with mouths open through patches of plankton',
             'Vocalizations include low-frequency upcalls, moans, and groans',
             'Callosities or raised skin patches are unique fingerprints',
             'Make upcalls to say "hello" to other whales nearby',
             'Use vocalizations for warnings, aggression, social bonding'];

// DOM Variables
let hearDiv1;
let hearDiv2;
let explanationDiv1;
let explanationDiv2;
let explanationDiv3;
let explanationDiv4;
let explanationDiv5;
let chooseDiv1;
let chooseDiv2;
let correctDiv;
let button0;
let button1;
let button2;
let button3;
let sel;
let ctxOn;

// Variables for Whale and WhaleSystem class
let system;
let all_calls;
let real1;
let real2;
let real3;
let fake4;
let num_real_calls = 3;
let calls = [];
let clock_moveXY = 0;
let clock_max = 500;
let clock_min = 300;

// Variables for SampleRNN neural network model
// SampleRNN is an end-to-end unconditional neural network for synthesizing raw audio
// waveforms. A hierarchy of RNN modules are used to model long-term dependencies
// at different temporal scales, with each module conditioning the one below it so that 
// the lowest module outputs sample-by-sample predictions. Each prediction is a q-way
// softmax over quantized values of the audio waveform.
// The goal of this app is to generate a 2-second long audio waveform of a right 
// whale upcall. Given a sampling rate of 2000 samples/second, the "samples" vector
// will need to contain 4000 samples. A linear quantization with q=256 will be used
// so that each sample of the "samples" vector will be one of 256 possible discrete
// values of the audio waveform. 
let srnn;
// A Three-Tier SampleRNN will be used, where "slow_predictor_final" is the highest
// level RNN module in the hierarchy processing 8 samples at a time, 
// "mid_predictor_final" is the second-highest RNN module in the hierarchy processing 
// 2 samples at a time, and "top_predictor_final" is the lowest module in the 
// hierarchy outputting sample predictions.
let slow_predictor;
let mid_predictor;
let top_predictor;
let slow_predictor_final;
let mid_predictor_final;
let top_predictor_final;
// Variables for the weights for each layer of SampleRNN
let slow_rnn_h0;
let mid_rnn_h0;
let slow_rnn_h_0;
let mid_rnn_h_0;
let slow_rnn_h;
let mid_rnn_h;
let slow_pred_rnn_0;
let slow_pred_rnn_1;
let slow_pred_rnn_2;
let slow_pred_rnn;
let slow_rnn;
let slow_td_0;
let slow_td_1;
let slow_td;
let mid_tdpre_0;
let mid_tdpre_1;
let mid_tdpre;
let mid_pred_rnn_0;
let mid_pred_rnn_1;
let mid_pred_rnn_2;
let mid_pred_rnn;
let mid_rnn;
let mid_tdpost_0;
let mid_tdpost_1;
let mid_tdpost;
let top_embed_0;
let top_embed;
let top_tdpre_0;
let top_tdpre;
let top_td1_0;
let top_td1_1;
let top_td1;
let top_td2_0;
let top_td2_1;
let top_td2;
let top_td3_0;
let top_td3_1;
let top_td3;
// Batch size
let batch_size = 468; 
// Random Seed
let random_seed = 1999;
// The trained SampleRNN is tasked with synthesizing a new 2-second long audio clip
// time series vector of a right whale upcall. 
// "samples" is an array holding the samples of the 2-second long time series vector
// "ts" is the maximum number of elements in the "samples" vector
let ts = 4000;
// "slow_fs" is the length of a frame processed by "slow_predictor_final," the highest
// level module in the SampleRNN hierarchy. In other words, "slow_predictor_final"
// processes 8-sample-long frames at a time.
let slow_fs = 8;
// "mid_fs" is the length of a frame processed by "mid_predictor_final," the second 
// highest level module in the SampleRNN hierarchy. In other words, 
// "mid_predictor_final" processes 2-sample-long frames at a time.
let mid_fs = 2;
// The number of possible quantized levels
let q_levels = 256;
// The trained SampleRNN is tasked with synthesizing a new 2-second long right whale
// upcall, which will be contained in the array "samples." As noted previously, the
// model can only make predictions from the 9th sample onwards. Therefore, the first
// 8 samples must be chosen a priori. In this case, the values of the first 8 samples
// from one of the examples in the training set is used as the first 8 samples for
// the vector "samples."
let samples = (arr=[]);
samples.length = ts;
samples.fill(0);
samples[0] = 49;
samples[1] = 28;
samples[2] = 126;
samples[3] = 126;
samples[4] = 56;
samples[5] = 255;
samples[6] = 211;
samples[7] = 145;

// Alternatively, the values of the first 8 samples could be chosen randomly from
// the set of possible quantized levels 
// for (var ii=0;ii<slow_fs;ii++) {
//     // Math.floor(Math.random() * (max - min + 1)) + min;
//     temp = Math.floor(Math.random() * (q_levels - 0)) + 0;
//     samples[ii] = temp;
// }

// "big_frame_level_outputs" is the conditioning vector from "slow_predictor_final"
// given to "mid_predictor_final"
let big_frame_level_outputs;
// "frame_level_outputs" is the conditioning vector from "mid_predictor_final" given
// to "top_predictor_final"
let frame_level_outputs;
// The variable "t" is used as an index into the "samples" vector. When
// "top_predictor_final" makes a new sample prediction, the value of the sample
// prediction is set as the value of the corresponding element in the "samples"
// vector. "t" is incremented by 1 each time a new prediction is made, so that the
// "samples" vector is gradually filled with all the predictions for the samples. 
// Note that "t" is initialized as 8 because the first 8 samples are given a priori 
// and the neural network is tasked with predicting the 9th sample onward (indexed
// as 8 because Javascript uses zero-based indexing). 
let t = 8;
// The variable "complete" indicates whether or not the SampleRNN has finished 
// predicting samples for the "samples" vector
let complete = 0;
// Due to the Web Audio API not accepting a sample rate of 2000 samples/second, the
// completed "samples" vector is upsampled by 2 and the result stored in "samples_up.""
let samples_up;
// After upsampling, in order to preserve the range of frequencies in the original 
// time series, the upsampled time series is low pass filtered and the result stored
// in "samples_up_filt."
let samples_up_filt;
// In order for the Web Audio API to translate the values of the time series into 
// sound, the values must be normalized. "samples_new" is the normalized version of
// "samples_up_filt."
let samples_new;

/**
 * Helper function to convert a JSON file into a 2D array
 * @param json JSON file
 */
function convertJSONto2D(json) {
  let d2Array = [];
  let result  = [];

  for(key in json) {
      result.push(json[key]);
  }
  d2Array.push(result);
  return d2Array[0];
}      

/**
 * Helper function to find the index of the largest element in an array
 * @param array Array
 */
function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

/**
 * Helper function to reshape a 1D array into a 2D array with the specified number
 * of rows and columns
 * @param orig_array Array
 * @param rows Number of desired rows in the output 2D array
 * @param cols Number of desired columns in the output 2D array
 */
function reshape(orig_array, rows, cols) {
  let new_array = [];
  let copy = orig_array.slice(0); // Copy all elements.

  for (let r = 0; r < rows; r++) {
    let row = [];
    for (let c = 0; c < cols; c++) {
      let i = r * cols + c;
      if (i < copy.length) {
        row.push(copy[i]);
      }
    }
    new_array.push(row);
  }
  return new_array;
}

// After the trained SampleRNN finishes loading, the SampleRNN will be tasked with 
// synthesizing a new 2-second long right whale upcall sound. As this will take time,
// a progress bar will be displayed on the web page to indicate the percentavge of
// samples that have been predicted (using the current value of "t" to calculate 
// the percentage displayed).
/**
 * Function to display progress bar
 */
function bar_move() {
  let elem = document.getElementById("myBar");
  let width = Math.round(t*0.025);
  elem.style.width = width + '%';
  elem.innerHTML = width*1+'%';
}

//========= Whale  ===========

// Four "whales" will be drawn to the canvas, three of which will make "real" right
// whale upcall sounds from the dataset. The fourth will make a "fake" right whale
// upcall sound synthesized by the SampleRNN. The task of the viewer will be to
// identify the whale that made the "fake" right whale upcall sound.
/**
 * Class for Whales
 * Based on code from: https://p5js.org/examples/simulate-soft-body.html
 * @constructor
 */
let Whale= function() {
  // Center Point for Whales initialized to a random location on the canvas
  this.centerX = Math.floor(Math.random() * (width));
  this.centerY = Math.floor(Math.random() * (height));
  // Previous x and y values of the center point, used to keep track of the movement
  // of the center point.
  this.prevcenterX = 0;
  this.prevcenterY = 0;
  // Whales will grow in size to "fullradius" and will start out on the canvas with
  // size "this.radius." 
  this.fullradius = Math.floor(Math.random() * (50 - 40 + 1)) + 40;
  this.radius = Math.floor(Math.random() * (15 - 10 + 1)) + 10;
  // Current difference between the full radius and current radius
  this.diffradius = this.fullradius-this.radius;
  // Variables for movement
  this.rotAngle = 0;
  this.accelX = 0.0;
  this.accelY = 0.0;
  this.deltaX = 0.0;
  this.deltaY = 0.0;
  this.springing = 0.0002;
  this.damping = 0.90;

  // Number of Corner Nodes
  this.nodes = 5;

  // Arrays keeping track of the location of the nodes of the shape defining the 
  // Whale 
  this.nodeStartX = [];
  this.nodeStartY = [];
  this.nodeX = [];
  this.nodeY = [];
  this.angle = [];
  this.frequency = [];

  // Soft-body dynamics
  this.organicConstant = 1.0;

  // Variable for the sound the Whale can make
  this.call;
  // Variable used to pan the sound of the Whale
  this.panning = 0;
  // x and y value sof the point on the canvas the Whale will move toward
  this.moveX = Math.floor(Math.random() * (width));
  this.moveY = Math.floor(Math.random() * (height));
  // Additional code for 3D audio
  // this.panner1 = new p5.Panner3D();
  // this.call.disconnect();
  // this.call.connect(panner1);
  // this.mul = 2;

  // Initialize arrays to 0
  for (let i=0; i<this.nodes; i++){
    this.nodeStartX[i] = 0;
    this.nodeStartY[i] = 0;
    this.nodeX[i] = 0;
    this.nodeY[i] = 0;
    this.angle[i] = 0;
  }

  // Iniitalize frequencies for corner nodes
  for (let i=0; i<this.nodes; i++){
    this.frequency[i] = random(17, 22);
  }
};

// Once the trained SampleRNN is finished loading and has begun synthesizing the
// new right whale upcall sound, four "baby" Whales will be created on the canvas,
// each with randomly calculated initial radius. The "growing" function will increase
// the radius of the whales based on the current value of "t" until the Whales reach
// their respective "full radius" values. 
/**
 * Function to increase size of Whale over time
 * @param t_value Current value of "t", indicating how much progress the SampleRNN
 *                has made in synthesizing the "fake" right whale upcall sound
 */
Whale.prototype.growing = function(t_value) {
  let temp_radius = (t_value*0.00025)*this.fullradius;
  if (temp_radius < this.radius) {
    this.radius = this.radius;
  } else {
    this.radius = temp_radius;
  }
}

/**
 * Function to get the x coordinate of the Whale's center point
 */
Whale.prototype.xcoords = function() {
  return this.centerX;
}

// The three "real" right whale upcall sounds will be mp3 files and loaded via 
// p5.js' loadSound() function. Save these sounds in the "call" variable for 
// the appropriate Whale using the addCall() method.
/**
 * Function to save sound file of call to the corresponding Whale
 * @param call Sound file of "real" right whale upcall
 */
Whale.prototype.addCall = function(call) {
  this.call = call;
}

/**
 * As the Whales move around the canvas, randomly and gradually change their rotation
 * angle using changeRot().
 */
Whale.prototype.changeRot = function() {
  // Calculate dX and dY using the location of the current center point and previous
  // center point. 
  let dX = this.centerX - this.prevcenterX;
  let dY = this.centerY - this.prevcenterY;
  // Determine the angle in which the Whale is moving using Math.atan2() and the 
  // calculated values of dX and dY
  let angle_rad = Math.atan2(dY,dX);
  let angle_deg = angle_rad * (180/Math.PI);
  // Find the difference between the current angle and previous angle the Whale
  // was moving
  let angle_diff = angle_deg - this.rotAngle; 
  // If the Whale's current angle is greater than its previous angle, rotate it 
  // slightly to the left. On the other hand if the Whale's current angle is less
  // than its previous angle, rotate it slightly to the right.
  if (angle_diff > 0) {
    this.rotAngle = this.rotAngle + Math.floor(Math.random() * (2 - 1 + 1)) + 1;
  } else if (angle_diff < 0) {
    this.rotAngle = this.rotAngle - Math.floor(Math.random() * (2 - 1 + 1)) + 1;
  }
}

/**
 * Function to change the Whale's "moveX" and "moveY" variables (the x and y values
 * of the location on the canvas the Whale will move toward).
 */
Whale.prototype.moveXY = function() {
  // Randomly determine the new values of the "moveX" and "moveY" variables.
  let rand_indX = Math.random();
  // For the majority of the time, set "moveX" and "moveY" to the outer edges of the 
  // canvas so that the Whale is encouraged to traverse larger distances across the
  // space. Occassionally, set the location corresponding to the "moveX" and "moveY" 
  // variables to the middle of the canvas.
  if (rand_indX > 0.65) {
    this.moveX = Math.floor(Math.random() * (width - (width-100) + 1)) + (width-100);
  } else if (rand_indX < 0.35) {
    this.moveX = Math.floor(Math.random() * (100 - 0 + 1)) + 0;
  } else if (rand_indX >= 0.35 && rand_indX <= 0.65) {
    this.moveX = Math.floor(Math.random() * ((width-100) - 100 + 1)) + 100;
  }
  let rand_indY = Math.random();
  if (rand_indY > 0.65) {
    this.moveY = Math.floor(Math.random() * (width - (width-100) + 1)) + (width-100);
  } else if (rand_indY < 0.35) {
    this.moveY = Math.floor(Math.random() * (100 - 0 + 1)) + 0;
  } else if (rand_indY >= 0.35 && rand_indY <= 0.65) {
    this.moveY = Math.floor(Math.random() * ((width-100) - 100 + 1)) + 100;
  }
}

/**
 * Function to draw the shape defining the Whale
 */
Whale.prototype.drawShape = function() {
  //  calculate node starting locations
  for (var i=0; i<this.nodes; i++){
    this.nodeStartX[i] = this.centerX+cos(radians(this.rotAngle))*this.radius;
    this.nodeStartY[i] = this.centerY+sin(radians(this.rotAngle))*this.radius;
    this.rotAngle += 360.0/this.nodes;
  }

  // Draw polygon
  curveTightness(this.organicConstant);
  fill(255);
  beginShape();
  for (var i=0; i<this.nodes; i++){
    curveVertex(this.nodeX[i], this.nodeY[i]);
  }
  for (var i=0; i<this.nodes-1; i++){
    curveVertex(this.nodeX[i], this.nodeY[i]);
  }
  endShape(CLOSE);
}

/**
 * Function to move the Whale in the direction of the point defined by "moveX" and
 * "moveY".
 */
Whale.prototype.moveShape = function() {
  // Move center point
  this.deltaX = this.moveX - this.centerX;
  this.deltaY = this.moveY - this.centerY;

  // Create springing effect
  this.deltaX *= this.springing;
  this.deltaY *= this.springing;
  this.accelX += this.deltaX;
  this.accelY += this.deltaY;

  this.prevcenterX = this.centerX;
  this.prevcenterY = this.centerY; 

  // Move predator's center
  this.centerX += this.accelX;
  this.centerY += this.accelY;

  // Slow down springing
  this.accelX *= this.damping;
  this.accelY *= this.damping;

  // Change curve tightness
  this.organicConstant = 1-((abs(this.accelX)+abs(this.accelY))*0.1);

  // Move nodes
  for (var i=0; i<this.nodes; i++){
    this.nodeX[i] = this.nodeStartX[i]+sin(radians(this.angle[i]))*(this.accelX*2);
    this.nodeY[i] = this.nodeStartY[i]+sin(radians(this.angle[i]))*(this.accelY*2);
    this.angle[i] += this.frequency[i];
  }
  this.panning = map(this.centerX, 0., width,-1.0, 1.0);
}

/**
 * Function to play the whale's associated call sound with panning value determined
 * by the "panning" variable
 */
Whale.prototype.play = function() {
  this.call.pan(this.panning);
  this.call.play();
}

//========= System of Whales  ===========

/**
 * Class for System of Whales
 * @constructor
 */
let WhaleSystem = function() {
  // Array of Whales in the system
  this.whales = [];
  // A global clock will be incrementing in the background as the Whales are drawn on
  // the canvas. In addition, each Whale in the system will be associated with a
  // different time - when the clock reaches this time, the corresponding Whale will
  // change the direction of its movement. 
  this.clocks = [];
};

/**
 * Function to change the "moveX" and "moveY" variables for each Whale in the system
 * if the global clock has reached the designated time for the individual Whale.
 * @param clock Current value of global clock "clock_moveXY"
 */
WhaleSystem.prototype.changemoveXY = function(clock) {
  for (let i = 0; i < this.whales.length; i++) {
    let w = this.whales[i];
    let c = this.clocks[i];
    if (clock > c) {
      w.moveXY(); 
    }
  }
}

/**
 * Function to add a new Whale to the system and a designated time for the Whale to
 * change the direction of its movement.
 */
WhaleSystem.prototype.addWhale = function() {
  this.whales.push(new Whale());
  this.clocks.push(Math.floor(Math.random() * (clock_max - clock_min + 1)) + clock_min);
};

/**
 * Function to call continuously in p5.js' draw() function to animate the Whales' 
 * movements. For every Whale in the system, the function draws the Whale, moves the
 * Whale, changes its rotation, and increases its size. 
 * @param t_value Current value of "t", indicating how much progress the SampleRNN
 *                has made in synthesizing the "fake" right whale upcall sound
 */
WhaleSystem.prototype.run = function(t_value) {
  for (let i = 0; i < this.whales.length; i++) {
    let w = this.whales[i];
    w.drawShape();
    w.moveShape();
    w.changeRot();
    w.growing(t_value);
  }
};

/**
 * Given an input array of three call sounds, this function associates each sound
 * with one of the first three Whales in the system. (The fourth and last Whale
 * will make the "fake" right whale upcall sound).
 * @param calls Array of "real" call sound files
 */
WhaleSystem.prototype.addCalls = function(calls) {
  // for (let i = 0; i < this.whales.length-1; i++) {
  //   let w = this.whales[i];
  //   w.addCall(calls[i]);
  for (let i = 0; i < this.whales.length; i++) {
    let w = this.whales[i];
    w.addCall(calls[i]);
  }
}

/**
 * Function to get the x coordinate of the center point for the desired Whale
 * @param whale_num Index of the desired Whale in the "whales" array of the system
 */
WhaleSystem.prototype.getxcoords = function(whale_num) {
  return this.whales[whale_num].xcoords();
}

/**
 * Function to play the call sound of the desired Whale
 * @param whale_num Index of the desired Whale in the "whales" array of the system
 */
WhaleSystem.prototype.makeCall = function(whale_num) {
  this.whales[whale_num].play();
}

/**
 * Function to play the call sound for the specified whale and set the variable
 * "isPlaying" to indicate whether or not a call sound is currently playing.
 * "concert" function of WhaleSystem uses the setting of "isPlaying" to ensure
 * that only one call sound is being played at once
 * @param whale_num Index of the desired Whale in the "whales" array of the system
 */
WhaleSystem.prototype.setPlaying = function(whale_num) {
  this.whales[whale_num].play();
  isPlaying = 1;
  setTimeout(() => {
    isPlaying = 0;
  }, 2000);
}

/**
 * Function to play the Whales' call sounds if they are close to each other (since
 * the right whale upcall is used by right whales in nature to greet each other).
 */
WhaleSystem.prototype.concert = function() {
  // Array to hold the locations of the center points for each Whale
  let proximity = []; 
  // Minimum distance whales must be from each other to start calling
  let min_dist = 10;
  // Calculate the distance of each Whale's center point from the origin and add it
  // to the "proximity" array. 
  for (let i = 0; i < this.whales.length; i++) {
    let w = this.whales[i];
    proximity.push(Math.sqrt(Math.pow(w.centerX,2)+Math.pow(w.centerY,2)));
  }
  // If two Whales are close in proximity to each other, have one of them play
  // its call sound.
  if ((Math.abs(proximity[0] - proximity[1]) < min_dist) && isPlaying == 0) {
    this.setPlaying(0);
  } else if ((Math.abs(proximity[0] - proximity[2]) < min_dist) && isPlaying == 0) {
    this.setPlaying(2);
  } else if ((Math.abs(proximity[0] - proximity[3]) < min_dist) && isPlaying == 0) {
    this.setPlaying(3);
  } else if ((Math.abs(proximity[1] - proximity[2]) < min_dist) && isPlaying == 0) {
    this.setPlaying(1);
  } else if ((Math.abs(proximity[1] - proximity[3]) < min_dist) && isPlaying == 0) {
    this.setPlaying(3);
  } else if ((Math.abs(proximity[2] - proximity[3]) < min_dist) && isPlaying == 0){
    this.setPlaying(2);
  }
}

/**
 * Softmax activation function applied to input array
 * @param arr Array
 */
function numpy_softmax(arr) {
  return arr.map(function(value,index) { 
    return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
  })
}

/**
 * Function treats the input array of probability predictions (for each of the possible
 * quantized levels) as a multinomial distribution and draws a sample from it
 * based on the provided random seed
 * @param coeff Array of probability predictions for the quantized levels
 * @param random_seed Desired random seed for sampling from the distribution
 */
function numpy_sample_softmax2d(coeff, random_seed) {
  // Convert the input array into a tensor
  let coeff_tensor = tf.tensor1d(coeff);
  // Use tf.multinomial() to draw a sample from the provided multinomial distribution
  // based on the given random seed
  let idxs = tf.multinomial(coeff_tensor,1,seed=random_seed,normalized=true).dataSync()[0];
  // let idxs = argMax(Array.from(coeff));
  return idxs;
}

/**
 * Function to call numpy_sample_softmax2d()
 * @param logits Array of probability predictions for the quantized levels
 * @param random_seed Desired random seed for sampling from the distribution
 */
function numpy_sample_softmax(logits, random_seed) {
  sample = numpy_sample_softmax2d(logits, random_seed);
  return sample;
}

/**
 * Function to make sample-by-sample predictions for the "samples" vector
 */
async function pred_samples() {
  // Obtain a new conditioning vector from "slow_predictor_final" every "slow_fs" = 8
  // samples
  if (t % slow_fs == 0) {
      big_frame_level_outputs = slow_predictor_final.predict(tf.tensor1d(samples.slice(t-slow_fs,t)).reshape([1,8,1])).dataSync();
      big_frame_level_outputs = reshape(big_frame_level_outputs,4,q_levels);
  }
  // Obtain a new conditioning vector from "mid_predictor_final" every "mid_fs"  = 2
  // samples
  if (t % mid_fs == 0) {
      temp_ind = Math.floor((t/mid_fs)%(slow_fs/mid_fs))
      frame_level_outputs = mid_predictor_final.predict([tf.tensor1d(samples.slice(t-mid_fs,t)).reshape([1,2,1]),tf.tensor1d(big_frame_level_outputs.slice(temp_ind,temp_ind+1)[0]).reshape([1,1,q_levels])]).dataSync();
      frame_level_outputs = reshape(frame_level_outputs,2,q_levels);
  }
  // Obtain the raw, unnormalized predictions from "top_predictor_final" based on the
  // conditioning vector from "mid_predictor_final"
  temp_ind = Math.floor(t%mid_fs)
  sample_prob = top_predictor_final.predict([tf.tensor1d(samples.slice(t-1,t)).reshape([1,1,1]),tf.tensor1d(frame_level_outputs.slice(temp_ind,temp_ind+1)[0]).reshape([1,1,q_levels])]).dataSync();
  // Apply the softmax activation function to the raw, unnormalized predictions
  sample_prob = numpy_softmax(sample_prob);
  // Draw a sample from the multinomial distribution (which will represent a new chosen
  // quantized level for the "samples" vector) and assign the value to the appropriate
  // element in the "samples" vector
  samples[t] = numpy_sample_softmax(sample_prob, random_seed);
  await big_frame_level_outputs;
  await frame_level_outputs;
  await sample_prob;
}

/**
 * Function to upsample the "samples" vector by a factor of 2 by inserting a zero
 * between each element so that the output "samples_up" vector is twice as long
 * as the original
 * @param sig original "samples" vector
 */
function upsample_2(sig) {
  let sig_up = (arr=[]);
  sig_up.length = sig.length*2;
  sig_up.fill(0);
  for (var ii = 0; ii < sig_up.length; ii+=2) {
      sig_up[ii] = sig[ii/2];
  }
  return sig_up; 
}

/**
 * Alternative function to upsample the "samples" vector by a factor of 2 by 
 * linearly interpolating between samples
 * @param sig original "samples" vector
 */
function linear_2(sig) {
  let sig_up = (arr=[]);
  sig_up.length = sig.length*2;
  sig_up.fill(0);
  for (var ii = 0; ii < sig_up.length; ii+=2) {
      sig_up[ii] = sig[ii/2];
  }
  for (var jj = 1; jj < sig_up.length; jj+=2) {
      if (jj==sig_up.length-1) {
          sig_up[jj] = sig_up[jj-1];
      } else {
      sig_up[jj] = (sig_up[jj-1] + sig_up[jj+1])/2;
      }
  }
  return sig_up; 
}

// DSP Filter Implementation in Javascript
// Based on code from: https://github.com/markert/fili.js/
  /**
   * Fir filter
   */
var FirFilter = function (filter) {
    // note: coefficients are equal to input response
    var f = filter
    var b = []
    var cnt = 0
    for (cnt = 0; cnt < f.length; cnt++) {
      b[cnt] = {
        re: f[cnt],
        im: 0
      }
    }
  
    var initZero = function (cnt) {
      var r = []
      var i
      for (i = 0; i < cnt; i++) {
        r.push(0)
      }
      return {
        buf: r,
        pointer: 0
      }
    }
  
    var z = initZero(f.length - 1)
  
    var doStep = function (input, d) {
      d.buf[d.pointer] = input
      var out = 0
      for (cnt = 0; cnt < d.buf.length; cnt++) {
        out += (f[cnt] * d.buf[(d.pointer + cnt) % d.buf.length])
      }
      d.pointer = (d.pointer + 1) % (d.buf.length)
      return out
    }
  
    var calcInputResponse = function (input) {
      var tempF = initZero(f.length - 1)
      return runMultiFilter(input, tempF, doStep)
    }
  
    var calcResponse = function (params) {
      var Fs = params.Fs
      var Fr = params.Fr
      // z = exp(j*omega*pi) = cos(omega*pi) + j*sin(omega*pi)
      // z^-1 = exp(-j*omega*pi)
      // omega is between 0 and 1. 1 is the Nyquist frequency.
      var theta = -Math.PI * (Fr / Fs) * 2
      var h = {
        re: 0,
        im: 0
      }
      for (var i = 0; i < f.length - 1; i++) {
        h = complex.add(h, complex.mul(b[i], {
          re: Math.cos(theta * i),
          im: Math.sin(theta * i)
        }))
      }
      var m = complex.magnitude(h)
      var res = {
        magnitude: m,
        phase: complex.phase(h),
        dBmagnitude: 20 * Math.log(m) * Math.LOG10E
      }
      return res
    }
  
    var self = {
      responsePoint: function (params) {
        return calcResponse(params)
      },
      response: function (resolution) {
        resolution = resolution || 100
        var res = []
        var cnt = 0
        var r = resolution * 2
        for (cnt = 0; cnt < resolution; cnt++) {
          res[cnt] = calcResponse({
            Fs: r,
            Fr: cnt
          })
        }
        evaluatePhase(res)
        return res
      },
      simulate: function (input) {
        return calcInputResponse(input)
      },
      singleStep: function (input) {
        return doStep(input, z)
      },
      multiStep: function (input, overwrite) {
        return runMultiFilter(input, z, doStep, overwrite)
      },
      filtfilt: function (input, overwrite) {
        return runMultiFilterReverse(runMultiFilter(
          input, z, doStep, overwrite), z, doStep, true)
      },
      reinit: function () {
        z = initZero(f.length - 1)
      }
    }
    return self
}

var FirCoeffs = function () {
    // Kaiser windowd filters
    // desired attenuation can be defined
    // better than windowd sinc filters
    var calcKImpulseResponse = function (params) {
      var Fs = params.Fs
      var Fa = params.Fa
      var Fb = params.Fb
      var o = params.order || 51
      var alpha = params.Att || 100
      var ino = function (val) {
        var d = 0
        var ds = 1
        var s = 1
        while (ds > s * 1e-6) {
          d += 2
          ds *= val * val / (d * d)
          s += ds
        }
        return s
      }
  
      if (o / 2 - Math.floor(o / 2) === 0) {
        o++
      }
      var Np = (o - 1) / 2
      var A = []
      var beta = 0
      var cnt = 0
      var inoBeta
      var ret = []
  
      A[0] = 2 * (Fb - Fa) / Fs
      for (cnt = 1; cnt <= Np; cnt++) {
        A[cnt] = (Math.sin(2 * cnt * Math.PI * Fb / Fs) - Math.sin(2 * cnt * Math.PI * Fa / Fs)) / (cnt * Math.PI)
      }
      // empirical coefficients
      if (alpha < 21) {
        beta = 0
      } else if (alpha > 50) {
        beta = 0.1102 * (alpha - 8.7)
      } else {
        beta = 0.5842 * Math.pow((alpha - 21), 0.4) + 0.07886 * (alpha - 21)
      }
  
      inoBeta = ino(beta)
      for (cnt = 0; cnt <= Np; cnt++) {
        ret[Np + cnt] = A[cnt] * ino(beta * Math.sqrt(1 - (cnt * cnt / (Np * Np)))) / inoBeta
      }
      for (cnt = 0; cnt < Np; cnt++) {
        ret[cnt] = ret[o - 1 - cnt]
      }
      return ret
    }
  
    // note: coefficients are equal to impulse response
    // windowd sinc filter
    var calcImpulseResponse = function (params) {
      var Fs = params.Fs
      var Fc = params.Fc
      var o = params.order
      var omega = 2 * Math.PI * Fc / Fs
      var cnt = 0
      var dc = 0
      var ret = []
      // sinc function is considered to be
      // the ideal impulse response
      // do an idft and use Hamming window afterwards
      for (cnt = 0; cnt <= o; cnt++) {
        if (cnt - o / 2 === 0) {
          ret[cnt] = omega
        } else {
          ret[cnt] = Math.sin(omega * (cnt - o / 2)) / (cnt - o / 2)
          // Hamming window
          ret[cnt] *= (0.54 - 0.46 * Math.cos(2 * Math.PI * cnt / o))
        }
        dc = dc + ret[cnt]
      }
      // normalize
      for (cnt = 0; cnt <= o; cnt++) {
        ret[cnt] /= dc
      }
      return ret
    }
    // invert for highpass from lowpass
    var invert = function (h) {
      var cnt
      for (cnt = 0; cnt < h.length; cnt++) {
        h[cnt] = -h[cnt]
      }
      h[(h.length - 1) / 2]++
      return h
    }
    var bs = function (params) {
      var lp = calcImpulseResponse({
        order: params.order,
        Fs: params.Fs,
        Fc: params.F2
      })
      var hp = invert(calcImpulseResponse({
        order: params.order,
        Fs: params.Fs,
        Fc: params.F1
      }))
      var out = []
      for (var i = 0; i < lp.length; i++) {
        out.push(lp[i] + hp[i])
      }
      return out
    }
    var self = {
      lowpass: function (params) {
        return calcImpulseResponse(params)
      },
      highpass: function (params) {
        return invert(calcImpulseResponse(params))
      },
      bandstop: function (params) {
        return bs(params)
      },
      bandpass: function (params) {
        return invert(bs(params))
      },
      kbFilter: function (params) {
        return calcKImpulseResponse(params)
      },
      available: function () {
        return ['lowpass', 'highpass', 'bandstop', 'bandpass', 'kbFilter']
      }
    }
    return self
}

evaluatePhase = function (res) {
  var xcnt = 0
  var cnt = 0
  var pi = Math.PI
  var tpi = 2 * pi
  var phase = []
  for (cnt = 0; cnt < res.length; cnt++) {
    phase.push(res[cnt].phase)
  }
  res[0].unwrappedPhase = res[0].phase
  res[0].groupDelay = 0
  // TODO: more sophisticated phase unwrapping needed
  for (cnt = 1; cnt < phase.length; cnt++) {
    var diff = phase[cnt] - phase[cnt - 1]
    if (diff > pi) {
      for (xcnt = cnt; xcnt < phase.length; xcnt++) {
        phase[xcnt] -= tpi
      }
    } else if (diff < -pi) {
      for (xcnt = cnt; xcnt < phase.length; xcnt++) {
        phase[xcnt] += tpi
      }
    }
    if (phase[cnt] < 0) {
      res[cnt].unwrappedPhase = -phase[cnt]
    } else {
      res[cnt].unwrappedPhase = phase[cnt]
    }

    res[cnt].phaseDelay = res[cnt].unwrappedPhase / (cnt / res.length)
    res[cnt].groupDelay = (res[cnt].unwrappedPhase - res[cnt - 1].unwrappedPhase) / (pi / res.length)
    if (res[cnt].groupDelay < 0) {
      res[cnt].groupDelay = -res[cnt].groupDelay
    }
  }
  if (res[0].magnitude !== 0) {
    res[0].phaseDelay = res[1].phaseDelay
    res[0].groupDelay = res[1].groupDelay
  } else {
    res[0].phaseDelay = res[2].phaseDelay
    res[0].groupDelay = res[2].groupDelay
    res[1].phaseDelay = res[2].phaseDelay
    res[1].groupDelay = res[2].groupDelay
  }
}

runMultiFilter = function (input, d, doStep, overwrite) {
  var out = []
  if (overwrite) {
    out = input
  }
  var i
  for (i = 0; i < input.length; i++) {
    out[i] = doStep(input[i], d)
  }
  return out
}

runMultiFilterReverse = function (input, d, doStep, overwrite) {
  var out = []
  if (overwrite) {
    out = input
  }
  var i
  for (i = input.length - 1; i >= 0; i--) {
    out[i] = doStep(input[i], d)
  }
  return out
}

var factorial = function (n, a) {
  if (!a) {
    a = 1
  }
  if (n !== Math.floor(n) || a !== Math.floor(a)) {
    return 1
  }
  if (n === 0 || n === 1) {
    return a
  } else {
    return factorial(n - 1, a * n)
  }
}

besselFactors = function (n) {
  var res = []
  for (var k = 0; k < n + 1; k++) {
    var p = factorial(2 * n - k)
    var q = Math.pow(2, (n - k)) * factorial(k) * factorial(n - k)
    res.unshift(Math.floor(p / q))
  }
  return res
}

var fractionToFp = function (fraction, fractionBits) {
  var fpFraction = 0
  for (var cnt = 0; cnt < fractionBits; cnt++) {
    var bitVal = 1 / Math.pow(2, cnt + 1)
    if (fraction > bitVal) {
      fraction -= bitVal
      fpFraction += bitVal
    }
  }
  return fpFraction
}

var numberToFp = function (number, numberBits) {
  return number & Math.pow(2, numberBits)
}

var valueToFp = function (value, numberBits, fractionBits) {
  var number = Math.abs(value)
  var fraction = value - number
  var fpNumber = {
    number: numberToFp(number, numberBits).toString(),
    fraction: fractionToFp(fraction, fractionBits).toString(),
    numberBits: numberBits,
    fractionBits: fractionBits
  }
  return fpNumber
}

fixedPoint = {
  convert: function (value, numberBits, fractionBits) {
    return valueToFp(value, numberBits, fractionBits)
  },
  add: function (fpVal1, fpVal2) {
  },
  sub: function (fpVal1, fpVal2) {
  },
  mul: function (fpVal1, fpVal2) {
  },
  div: function (fpVal1, fpVal2) {
  }
}

complex = {

  div: function (p, q) {
    var a = p.re
    var b = p.im
    var c = q.re
    var d = q.im
    var n = (c * c + d * d)
    var x = {
      re: (a * c + b * d) / n,
      im: (b * c - a * d) / n
    }
    return x
  },
  mul: function (p, q) {
    var a = p.re
    var b = p.im
    var c = q.re
    var d = q.im
    var x = {
      re: (a * c - b * d),
      im: (a + b) * (c + d) - a * c - b * d
    }
    return x
  },
  add: function (p, q) {
    var x = {
      re: p.re + q.re,
      im: p.im + q.im
    }
    return x
  },
  sub: function (p, q) {
    var x = {
      re: p.re - q.re,
      im: p.im - q.im
    }
    return x
  },
  phase: function (n) {
    return Math.atan2(n.im, n.re)
  },
  magnitude: function (n) {
    return Math.sqrt(n.re * n.re + n.im * n.im)
  }
}

/**
 * Function to convert the "samples_up_filt" vector into a JS ArrayBuffer (raw binary
 * data) that can be played using the Web Audio API
 * @param samples_up_filt Filtered, upsampled version of original "samples" vector
 */
function arr_to_wav(samples_up_filt) {
  // Normalize the input "samples_up_filt" vector between -0.5 and 0.5
  let samples_max = Math.max.apply(null, samples_up_filt.map(Math.abs));
  let samples_min = Math.min.apply(null, samples_up_filt.map(Math.abs));
  samples_new = samples_up_filt.map(function(x) { return x - samples_min; });
  samples_new = samples_new.map(function(x) { return x / samples_max; });
  samples_new = samples_new.map(function(x) { return x - 0.5; });
  samples_new = samples_new.map(function(x) { return x * 0.95; });
  // Create an empty two-second mono buffer at the sample rate 4000 samples/second
  // (twice the original sample rate since "samples_new is twice as long as the 
  // original "samples" vector)
  myArrayBuffer = audioCtx.createBuffer(1, 4000*2, 4000);
  // This gives us the actual array that contains the data
  for (let channel = 0; channel < myArrayBuffer.numberOfChannels; channel++) {
      let nowBuffering = myArrayBuffer.getChannelData(channel);
      // Fill the array with the values from "samples_new"
      for (let m = 0; m < myArrayBuffer.length; m++) {
          nowBuffering[m] = samples_new[m];
      }
  }
}

/**
 * Function to add the loaded mp3 files of the "real" right whale upcall sounds into
 * the "calls" array
 * @param num_real_calls Number of "real" call sounds to be used
 */
function collect_calls(num_real_calls) {
  // for (var i = 0; i < num_real_calls; i++) {
  //   let call_ind = Math.floor(Math.random() * (num_real_calls));
  //   calls.push(all_calls[call_ind]);
  for (var i = 0; i < num_real_calls+1; i++) {
    calls.push(all_calls[i]);
  }
}

/**
 * Function to create the "fake" right whale upcall sound once the SampleRNN has 
 * finished synthesizing and save it to the AudioContext's buffer
 */
function create_fake_call() {
  // Upsample the "samples" vector by a factor of 2
  samples_up = upsample_2(samples);
  // Use the DSP Filter code from fili.js to low-pass filter the "samples_up" vector
  // (thereby ensuring that the original range of frequencies is present in the 
  // new, upsampled time series vector).
  var firCalculator = new FirCoeffs();
  var firFilterCoeffs = firCalculator.lowpass({
      order: 6, // Filter order
      Fs: 4000, // Sampling frequency
      Fc: 200 // Cutoff frequency
  });
  var firFilter = new FirFilter(firFilterCoeffs);
  samples_up_filt = firFilter.simulate(samples_up);
  // Save the values to the AudioContext's buffer
  arr_to_wav(samples_up_filt);
}  

/**
 * Function to play the "fake" right whale upcall sound
 * @param whale_num Index of the desired Whale in the "whales" array of the system
 */
function play_fake_call(whale_num) {
    // Get an AudioBufferSourceNode.
    // This is the AudioNode to use when we want to play an AudioBuffer
    let source = audioCtx.createBufferSource();
    // Additional parameters to change the directionality of the sound
    // for 3D audio
    // panner.coneOuterGain = 0.1;
    // panner.coneOuterAngle = 180;
    // panner.coneInnerAngle = 0;
    // audioCtx.listener.setPosition(0,0,0);
    // Set the buffer in the AudioBufferSourceNode
    source.buffer = myArrayBuffer;
    // Get the x coordinates of the center point of the Whale assigned to the "fake"
    // right whale upcall sound and use it to determine the sound's panning position
    let pan_value = map(system.getxcoords(whale_num), 0., width,-1.0, 1.0);
    panner.pan.value = pan_value;
    // Additional code if filtering is desired in the audio pipeline
    // var biquadFilter = audioCtx.createBiquadFilter();
    // sound.panner.setPosition(p.x, p.y, p.z);
    // Connect the AudioBufferSourceNode to the panner 
    source.connect(panner);
    // Additional connections if filtering is desired in the audio pipeline
    // biquadFilter.connect(panner);
    // biquadFilter.connect(audioCtx.destination);
    // biquadFilter.type = "lowpass";
    // biquadFilter.frequency.value = 200;
    // Connect the panner to the destination so we can hear the sound
    panner.connect(audioCtx.destination);
    // Start the source playing
    source.start();
}

/**
 * Function to load the trained SampleRNN from the server
 */
async function load_models() {
  console.log('Loading models');
  srnn = tf.loadModel('http://localhost:8080/srnn_fit/model.json',strict=false);
  srnn = await srnn;
  console.log('SRNN loaded');

  // The GRUs in "slow_predictor_final" and "mid_predictor_final" have learned initial
  // states that need to be manually set.
  // Convert the JSON files holding the initial states of the GRUs into 2D arrays
  // and use tf.tile() to convert them into the proper shape. (There is a state for 
  // each sample of a batch. Since the GRUs are "stateful," the last states of samples
  // in a batch are used as the initial states for samples in the next batch. In this
  // way, shorter subsequences from a longer sequence can be processed by recalling
  // the states from previous adjacent subsequences). 
  let slow_rnn_h_2d = convertJSONto2D(slow_rnn_h_0);
  slow_rnn_h = tf.tensor2d(slow_rnn_h_2d,[1,256]);
  slow_rnn_h0 = slow_rnn_h.tile([batch_size,1])
  let mid_rnn_h_2d = convertJSONto2D(mid_rnn_h_0);
  mid_rnn_h = tf.tensor2d(mid_rnn_h_2d,[1,256]);
  mid_rnn_h0 = mid_rnn_h.tile([batch_size,1])

  // Manually create the architectures of each RNN module in the hierarchy of the 
  // SampleRNN. This is necessary in order to manually set the initial states of the
  // GRUs 
  // Architecture of the "slow_predictor_final" module
  // "slow_predictor_final" uses a stateful GRU and dense layers to create a 
  // conditioning vector to pass to "mid_predictor_final." "slow_predictor_final"
  // receives as input 8 samples to process.
  slow_predictor = [];
  slow_predictor[0] = tf.input({batchShape:[1,8,1]});
  slow_predictor[1] = tf.layers.reshape({targetShape: [1,8]}).apply(slow_predictor[0]);
  slow_predictor[2] = tf.layers.gru({units:256,stateful:true,returnSequences:true,recurrentActivation:'sigmoid',initialState:slow_rnn_h0}).apply(slow_predictor[1]);
  slow_predictor[3] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 1024})}).apply(slow_predictor[2]);
  slow_predictor[4] = tf.layers.reshape({targetShape: [4,256]}).apply(slow_predictor[3]);
  slow_predictor_final = tf.model({inputs: slow_predictor[0], outputs: slow_predictor[4]});

  // Architecture of the "mid_predictor_final" module
  // "mid_predictor_final" uses a stateful GRU and dense layers to create a 
  // conditioning vector to pass to "top_predictor_final." "mid_predictor_final"
  // receives as input 2 samples to process as well as the conditioning vector from
  // "top_predictor_final." In order to add these two inputs together, a linear 
  // projection layer is used to ensure that the dimensions of the two inputs are
  // equal. 
  mid_predictor = [];
  mid_predictor[0] = tf.input({batchShape:[1,2,1]});
  mid_predictor[1] = tf.layers.reshape({targetShape: [1,2]}).apply(mid_predictor[0]);
  mid_predictor[2] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256})}).apply(mid_predictor[1]);
  mid_predictor[3] = tf.input({batchShape:[1,1,256]});
  mid_predictor[4] = tf.layers.add().apply([mid_predictor[2],mid_predictor[3]]);
  mid_predictor[5] = tf.layers.gru({units:256,stateful:true,returnSequences:true,recurrentActivation:'sigmoid',initialState:mid_rnn_h0}).apply(mid_predictor[4]);
  mid_predictor[6] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 512})}).apply(mid_predictor[5]);
  mid_predictor[7] = tf.layers.reshape({targetShape: [2,256]}).apply(mid_predictor[6]);
  mid_predictor_final = tf.model({inputs: [mid_predictor[0],mid_predictor[3]], outputs: mid_predictor[7]});

  // Architecture of the "top_predictor_final" module
  // "top_predictor_final" receives as input the previous (quantized) value of the 
  // audio clip time series and passes it into an embedding layer, which maps each
  // of the quantized values to a real-valued vector embedding. "top_predictor_final"
  // also receives as input the conditioning vector from "mid_predictor_final." In 
  // order to add these two inputs together, a linear projection layer is used. 
  // The output of the addition is then passed through a Multilayer Perceptron to yield 
  // the final (quantized) prediction for the next sample. 
  top_predictor = [];
  top_predictor[0] = tf.input({batchShape:[1,1,1]});
  top_predictor[1] = tf.layers.reshape({targetShape: [1]}).apply(top_predictor[0]);
  top_predictor[2] = tf.layers.embedding({inputDim: 256, outputDim: 256, embeddingsInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1.})}).apply(top_predictor[1]); 
  top_predictor[3] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256, useBias: false, kernelInitializer: 'leCunNormal'})}).apply(top_predictor[2]);
  top_predictor[4] = tf.input({batchShape:[1,1,256]});
  top_predictor[5] = tf.layers.add().apply([top_predictor[3],top_predictor[4]]);
  top_predictor[6] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256, activation: 'relu'})}).apply(top_predictor[5]);
  top_predictor[7] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256, activation: 'relu'})}).apply(top_predictor[6]);
  top_predictor[8] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256, kernelInitializer: 'leCunNormal'})}).apply(top_predictor[7]);
  top_predictor_final = tf.model({inputs: [top_predictor[0],top_predictor[4]], outputs: top_predictor[8]});

  // Set the weights of the three modules in the hierarchy of SampleRNN according 
  // to the corresponding trained weights.
  slow_predictor_final.layers[2].setWeights(srnn.layers[3].getWeights());
  slow_predictor_final.layers[3].setWeights(srnn.layers[5].getWeights());
  mid_predictor_final.layers[2].setWeights(srnn.layers[6].getWeights());
  mid_predictor_final.layers[5].setWeights(srnn.layers[10].getWeights());
  mid_predictor_final.layers[6].setWeights(srnn.layers[12].getWeights());
  top_predictor_final.layers[2].setWeights(srnn.layers[13].getWeights());
  top_predictor_final.layers[3].setWeights(srnn.layers[15].getWeights());
  top_predictor_final.layers[6].setWeights(srnn.layers[17].getWeights());
  top_predictor_final.layers[7].setWeights(srnn.layers[18].getWeights());
  top_predictor_final.layers[8].setWeights(srnn.layers[19].getWeights());

  // Set the "loading" variable to true, indicating that the model has finished
  // loading and the Whales can be drawn to canvas.
  loading = true;
}

/**
 * Callback function for "Select" element from p5.js
 */
function mySelectEvent() {
  fill(0);
  // If the correct Whale is chosen from the dropdown menu (as the whale producing
  // the "fake" right whale upcall), display "Correct!" Otherwise, display 
  // "Not this one".
  if (correctDiv) {
    correctDiv.remove();
  }
  var item = sel.value();
  if (item=='Willy') {
    correctDiv = createDiv('Correct!');
  } else {
    correctDiv = createDiv('Not this one');
  }
  correctDiv.style('color','black');
  correctDiv.style('position',width+20, 465);
}

/**
 * Preload the spectrogram image, "real" right whale upcall sounds, and JSON files
 * containing the initial states of the "slow_predictor_final" and 
 * "mid_predictor_final" GRUs.
 */
function preload() {
    // While the trained SampleRNN loads, a loading screen will be displayed on the
    // web page, showing a spectrogram of right whale upcalls. Preload this 
    // spectrogram image: 
    spectrogram = loadImage("spectrogram.png");
    // After the trained SampleRNN finishes loading, an ambient underwater sound will 
    // play in the background. The Whales will be drawn on the canvas and the SampleRNN
    // will begin synthesizing a new right whale upcall sound.
    soundFormats('mp3');
    ambient = loadSound('ambient.mp3');
    // There are eight "real" right whale upcall sounds available on the server 
    // to choose from. The "all_calls" vector will contain all these possible call
    // sounds.
    real1 = loadSound('real1.mp3');
    real2 = loadSound('real2.mp3');
    real3 = loadSound('real3.mp3');
    fake4 = loadSound('fake4.mp3');
    // Set the gain of the whale call sounds
    real1.setVolume(0.3);
    real2.setVolume(0.3);
    real3.setVolume(0.3);
    fake4.setVolume(0.3);
    all_calls = [real1,real2,real3,fake4];
    // Load the JSON files containing the initial states of the GRUs
    // var url = 'http://localhost:8080/slow_rnn_initialstate.json';
    // slow_rnn_h_0 = loadJSON(url);
    // url = 'http://localhost:8080/mid_rnn_initialstate.json';
    // mid_rnn_h_0 = loadJSON(url);
    loading = true;
}

function setup() {
    // Set the canvas size
    createCanvas(600,380);
    // Call the asynchronous load_models() function to begin loading the SampleRNN
    // load_models();
    // Set the gain of the ambient sound and set it to loop
    ambient.setVolume(0.3);
    // Additional code to use the 3D audio available from p5.js
    // panner1 = new p5.Panner3D();
    // ambient.disconnect();
    // ambient.connect(panner1);
    ambient.loop();
    // For the loading screen, choose a random fact from the "facts" array to display
    // using "text_ind".
    text_ind = Math.floor(Math.random() * (facts.length));
    noStroke();
    frameRate(200);
    // Create a new System of Whales
    system = new WhaleSystem();
    // Add four Whales to the system
    for (let ii = 0; ii < num_real_calls+1; ii++) {
      system.addWhale();
    } 
    // Create DOM variables
    // Create buttons for each Whale. If the user clicks a button, the corresponding
    // Whale's call will be played.
    button0= createButton('Migaloo');
    button0.mousePressed(function() {
      real1.setVolume(0.5);
      system.makeCall(0);
    });
    button1 = createButton('Keiko');
    button1.mousePressed(function() {
      real2.setVolume(0.5);
      system.makeCall(1);
    });
    button2 = createButton('Moby');
    button2.mousePressed(function() {
      real3.setVolume(0.5);
      system.makeCall(2);
    });
    button3 = createButton('Willy');
    button3.mousePressed(function() {
      fake4.setVolume(0.5);
      // play_fake_call(3);
      system.makeCall(3);
    });
    button0.position(width+20,285);
    button1.position(width+20,305);
    button2.position(width+20,325);
    button3.position(width+20,345);
    // Create a dropdown menu of the names of all the Whales. If a Whale name is
    // selected, use the mySelectEvent() callback.
    sel = createSelect();
    sel.position(width+20, 435);
    sel.option('Migaloo');
    sel.option('Keiko');
    sel.option('Moby');
    sel.option('Willy');
    sel.changed(mySelectEvent);
    // Button to turn on Audio
    let ctx = getAudioContext();
    ctxOn = createButton('Sound On');
    ctxOn.position(width+20,120);
    ctxOn.mousePressed(() => {
        ctx.resume().then(() => {
            ctxOn.hide();
  	    });
    });
    // Text to explain the DOM variables
    hearDiv1 = createDiv('Click the name of a whale');
    hearDiv2 = createDiv('to hear their call');
    hearDiv1.style('position',width+20,235);
    hearDiv2.style('position',width+20,255);
    chooseDiv1 = createDiv('Choose which whale you');
    chooseDiv2 = createDiv('think is the robot');
    chooseDiv1.style('position',width+20,385); 
    chooseDiv2.style('position',width+20,405); 
    explanationDiv1 = createDiv('Right whales make "upcalls"');
    explanationDiv2 = createDiv('to greet their neighbors.');
    explanationDiv3 = createDiv('But beware! One of them is');
    explanationDiv4 = createDiv('a robot pretending to be a');
    explanationDiv5 = createDiv('whale. Guess which one it is!');
    explanationDiv1.style('position',width+20,115); 
    explanationDiv2.style('position',width+20,135); 
    explanationDiv3.style('position',width+20,155); 
    explanationDiv4.style('position',width+20,175); 
    explanationDiv5.style('position',width+20,195); 
}

function draw() {
    if (loading) {
      // Code for after the SampleRNN has finished loading
      // Fade background
      fill(0, 100);
      rect(0,0,width,height);
      // Increment the global clock "clock_moveXY" by 1 every time draw() is called.
      // If "clock_moveXY" reaches "clock_max," reset the clock to 0.
      if (clock_moveXY >= clock_max) {
        clock_moveXY = 0;
      } else {
        clock_moveXY = clock_moveXY + 1;
      }
      // Call run() on the system
      system.run(t);
      // Call changemoveXY() on the system
      system.changemoveXY(clock_moveXY); 
      // Call bar_move() to update the progres bar, indicating how many samples 
      // the SampleRNN has predicted for the "samples" vector.
      bar_move();
      if (t >= 4000 && complete == 0) {
        // Code for when the "samples" vector is complete
        // Show DOM variables containing explanatory text
        explanationDiv1.show();
        explanationDiv2.show();
        explanationDiv3.show();
        explanationDiv4.show();
        explanationDiv5.show();
        hearDiv1.show();
        hearDiv2.show();
        chooseDiv1.show();
        chooseDiv2.show();
        button0.show();
        button1.show();
        button2.show();
        button3.show();
        sel.show();
        // Create the fake call using the "samples" vector
        // create_fake_call();
        // Add all the "real" calls to the Whales in the system
        collect_calls(num_real_calls);
        system.addCalls(calls);
        t = t + 1;
        complete = 1;
      } else if (t >= 4000 && complete == 1) {
        // Code for after the "samples" vector is complete
        // Show DOM variables containing explanatory text
        explanationDiv1.show();
        explanationDiv2.show();
        explanationDiv3.show();
        explanationDiv4.show();
        explanationDiv5.show();
        hearDiv1.show();
        hearDiv2.show();
        chooseDiv1.show();
        chooseDiv2.show();
        button0.show();
        button1.show();
        button2.show();
        button3.show();
        sel.show();
      // Call concert() on the system
      system.concert();
      } else {
          // Code for before the "samples" vector is complete
          // Hide the DOM variables containing explanatory text
          explanationDiv1.hide();
          explanationDiv2.hide();
          explanationDiv3.hide();
          explanationDiv4.hide();
          explanationDiv5.hide();
          hearDiv1.hide();
          hearDiv2.hide();
          chooseDiv1.hide();
          chooseDiv2.hide();
          button0.hide();
          button1.hide();
          button2.hide();
          button3.hide();
          sel.hide();
          // pred_samples();
          t = t+5;
      }
  } else {
    // Code for before the SampleRNN finishes loading (i.e. code for the loading
    // screen)
    // Hide the DOM variables containing explanatory text
    explanationDiv1.hide();
    explanationDiv2.hide();
    explanationDiv3.hide();
    explanationDiv4.hide();
    explanationDiv5.hide();
    hearDiv1.hide();
    hearDiv2.hide();
    chooseDiv1.hide();
    chooseDiv2.hide();
    button0.hide();
    button1.hide();
    button2.hide();
    button3.hide();
    sel.hide();
    // Display the spectrogram image
    image(spectrogram,0,0,width,height);
    // Display the text "Loading Whales..." along with the randomly chosen fact 
    // from the "facts" array
    textFont('Helvetica');
    textSize(16);
    fill(255);
    textAlign(LEFT,CENTER);
    text('Loading Whales...',2,20);
    text('Whale Fun Fact:',2,170)
    text(facts[text_ind],2,100,width,height/2);
  }
}
