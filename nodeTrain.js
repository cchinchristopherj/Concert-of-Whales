// Required modules
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const fs = require('fs');
const math = require('mathjs');

// Variables for SampleRNN neural network model
// SampleRNN is an end-to-end unconditional neural network for synthesizing raw audio
// waveforms. A hierarchy of RNN modules are used to model long-term dependencies
// at different temporal scales, with each module conditioning the one below it so that 
// the lowest module outputs sample-by-sample predictions. Each prediction is a q-way
// softmax over quantized values of the audio waveform.
// A Three-Tier SampleRNN will be used, where the highest level RNN module in the 
// hierarchy processes 8 samples at a time, the second-highest RNN module in the 
// hierarchy processes 2 samples at a time, and the lowest module in the hierarchy
// outputs sample predictions.
let srnn;
let srnn_final;
// "maxlen" is the maximum number of elements in an audio clip time series vector
let maxlen = 4000;
// The SampleRNN is trained using truncated backpropagation through time, in other 
// words the audio clip time series vector is split into shorter subsequences and 
// these subsequences are used for training (gradients are propagated to the beginning
// of these subsequences instead of across the entire time series vector).
// "cut_len" is the length of these shorter subsequences
let cut_len = 256;
// Variables for the weights for each layer of SampleRNN
let slow_rnn_h_0;
let mid_rnn_h_0;
let slow_rnn_h;
let mid_rnn_h;
let slow_rnn_h0;
let mid_rnn_h0;
let slow_pred_rnn_0;
let slow_pred_rnn_1;
let slow_pred_rnn_2;
let slow_pred_rnn;
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
// Optimizer for training
const optimizer = tf.train.adam();
// "slow_fs" is the length of a frame (8 samples) processed by the highest level module 
// in the SampleRNN hierarchy.
let slow_fs = 8;
// "mid_fs" is the length of a frame (2 samples) processed by the second highest level 
// module in the SampleRNN hierarchy.
let mid_fs = 2;
// The number of possible quantized levels
let q_levels = 256;
// During training of the SampleRNN, given the values of "slow_fs" and "mid_fs,"
// every new sample predicted by the loest module is conditioned on the previous
// 8 samples processed by the highest module and the previous 2 samples processed
// by the 2nd-highest module. Therefore, shorter subsequences for training via truncated
// backpropagation through time are created as follows: Each audio time series vector
// of a right whale upcall is split into subsequences of length 264 (which is equal
// to "cut_len"=256 plus "overlap"=8). The subsequences are 8 samples longer than 
// the desired length (256 samples) becuase 8 samples are needed to be processed 
// before making a prediction of a new sample. Therefore, everything but the last 8
// samples of the subsequence are used as input to the neural network and the ground
// truth is everything from the 9th sample to the end of the subsequence. 
// For example, the highest module uses the first 8 samples of the input to create
// a conditioning vector given to the 2nd-highest module.
// The 2nd-highest module uses the conditioning vector from the highest module
// along with the last two samples given to the highest module to create a 
// conditioning vector given to the lowest module.
// The lowest module uses the last sample given to the highest module to make 
// a prediction about the 9th sample in the time series. Therefore, since predictions
// are made from the 9th sample to the end of the time series, these samples compose
// the ground truth. In addition, since inputs are given from the beginning of the 
// time series up to 8 samples before the end of the time series (since these last 
// 8 samples cannot be used to make a prediction for a new sample), these samples 
// compose the input given to the neural network. Since the last 8 samples are always
// not included in each subsequence, the subsequences contain an overlap of 8 samples
// to ensure that every sample from a time series vector in the dataset is given 
// to the neural network for training. Note also that since everything from the 
// beginning of a time series up to the last 8 samples is given as input to the neural
// network, the subsequence length is equal to "cut_len"=256 as desired. 
let overlap = slow_fs;
// The dataset used to train the SampleRNN consists of 2-second long clips of right 
// whale upcalls sampled at 2000 samples/second that have been quantized into 256
// levels. "Y_train" is this dataset, while "X_train" is a version of this dataset 
// normalized between -2 and 2 (for input into the neural network). During training, 
// each audio clip time series vector from "X_train" is given as input to the 
// SampleRNN and the different modules in the hierarchy process different-length 
// frames from the time series vector. The lowest-level module in the hierarchy is
// tasked with making sample predictions and the ground truth (for computation of the
// loss function) is derived from relevant samples from the corresponding time series
// vector in "Y_train." 
let X_train;
let Y_train;

/**
 * Function to create shorter subsequences from an entire audio clip time series vector
 * for training via truncated backpropagation through time.
 * @param dataset Training set
 * @param ind Index of desired audio clip time series vector from the dataset
 * 
 */
function split_subsequences(dataset,ind) {
    // Calculate the total number of subsequences that need to be created
    let num_subsequences = (maxlen-cut_len)/overlap;
    let rows = num_subsequences;
    // Each subsequence is "cut_len" + "overlap" samples long
    let cols = cut_len+overlap;
    // Fill a new 2D matrix with the desired overlapped subsequences
    let dataset_new = Array.from(Array(rows), _ => Array(cols).fill(0));
    for (jj = 0; jj < num_subsequences; jj++) {
        dataset_new[jj] = dataset[ind].slice(jj*overlap,jj*overlap+cut_len+overlap);
    }
    return dataset_new;
}

/**
 * Custom "categorical_crossentropy" loss function for training
 * @param target Ground truth
 * @param output Prediction from model
 * 
 */
function categorical_crossentropy(target, output) {
    return tf.tidy(() => {
        target = target.asType("float32");
        output = output.asType("float32");
        let q_levels = 256;
        let new_target_shape = [output.shape[0],output.shape[1]];
        output = output.reshape([output.shape[0]*output.shape[1], q_levels]);
        let xdev = tf.sub(output,tf.max(output, axis=1, keepDims=true));
        xdev = xdev.asType("float32");
        let lsm = tf.sub(xdev,tf.log(tf.sum(tf.exp(xdev), axis=1, keepDims=true)).asType("float32"));
        lsm = lsm.asType("float32");
        target = target.reshape([target.shape[0]*target.shape[1], q_levels]);
        let cost = tf.mul(tf.sum(tf.mul(lsm,target),axis=1),tf.scalar(-1));
        cost = cost.asType("float32");
        let log2e = tf.scalar(math.log2(math.e));
        let temp_loss = tf.mul(cost.reshape(new_target_shape),log2e);
        return temp_loss;
    });
}

/**
 * Function to convert scalar class labels to one-hot vectors
 * @param labels_dense Scalar class labels
 * @param n_classes Number of possible classes
 * 
 */
function numpy_one_hot(labels_dense, n_classes) {
    return tf.tidy(() => {
        let num_samples = labels_dense.length;
        let num_timesteps = labels_dense[0].length;
        let temp1 = [];
        let temp2 = [];
        for (var ii=0; ii < num_samples; ii++) {
        for (var jj=0; jj < num_timesteps; jj++) {
            temp1.push(tf.oneHot(tf.tensor1d([labels_dense[ii][jj]],"int32"),n_classes).expandDims(axis=1));
        }
        temp2.push(tf.concat(temp1,1));
        temp1 = []; 
        }
        let temp3 = tf.concat(temp2,0);
        return temp3; 
    });
}

// Function that takes as input matrices of the desired subsequences of an entire audio 
// clip time series vector and splices them so that they are the appropriate shape and 
// contain the relevant samples for input to the three modules of the SampleRNN 
// hierarchy.
/**
 * Function to prepare batches for training
 * @param X_train Matrix of the desired subsequences with normalized values.
 * @param Y_train Matrix of the desired subsequences with unnormalized, quantized levels
 * 
 */
function _prep_batch(X_train,Y_train) {
    return tf.tidy(() => {
        // The highest module in the hierarchy processes 8 samples at a time and 
        // requires samples from the entire subsequence excluding the last 8 samples
        let x_slow = [];
        for(var i = 0; i < X_train.length; i++) {
        x_slow.push(X_train[i].slice(0,X_train[i].length-slow_fs));
        }
        x_slow = tf.tensor2d(x_slow);
        x_slow = x_slow.expandDims(axis=2);
        // The 2nd-highest module in the hierarchy processes 2 samples at a time. Recall
        // that the first prediction that is made is of the 9th sample. The 7th and 8th
        // samples are therefore used by the 2nd-highest module to make a prediction
        // for the 9th sample. The 2nd-highest module requires samples starting from the
        // 7th sample and ending two samples short of the end of the subsequence. 
        let x_mid = [];
        for(var i = 0; i < X_train.length; i++) {
        x_mid.push(X_train[i].slice(slow_fs-mid_fs,X_train[i].length-mid_fs));
        }
        x_mid = tf.tensor2d(x_mid);
        x_mid = x_mid.expandDims(axis=2);
        // The lowest module in the hierarchy processes 1 sample at a time (the previous
        // sample). Since the first prediction that is made is of the 9th sample, the
        // 8th sample is used by the lowest module to make a prediction for the 9th
        // sample. The lowest module requires samples starting from the 8th sample
        // and ending one sample short of the end of the subsequence. 
        let x_prev = [];
        for(var i = 0; i < Y_train.length; i++) {
        x_prev.push(Y_train[i].slice(slow_fs-1,Y_train[i].length-1));
        }
        x_prev = tf.tensor2d(x_prev);
        x_prev = x_prev.expandDims(axis=2);
        // The ground truth vector "target_train" requires samples starting from the
        // 9th sample to the end of the subsequence (since predictions are made 
        // starting from the 9th sample onward). 
        let target_train = [];
        for(var i = 0; i < Y_train.length; i++) {
        target_train.push(Y_train[i].slice(slow_fs,Y_train[i].length));
        }
        // Convert the scalar class label prediction to a one-hot vector
        target_train = numpy_one_hot(target_train, q_levels);
        return [x_slow, x_mid, x_prev, target_train];
    });
}

/**
 * Asynchronous function to train the neural network on one batch of the dataset 
 * @param X_train Matrix of the desired subsequences with normalized values.
 * @param Y_train Matrix of the desired subsequences with unnormalized, quantized levels
 * @param model Neural network model to train
 * 
 */
async function train_on_batch(X_train,Y_train,model) {
    // Save the appropriate matrices of subsequences of vectors to give to each of the
    // modules of the SampleRNN hierarchy in "temparr." Then unpack "temparr" into
    // the desired matrices "x_slow," "x_mid," "x_prev".
    let temparr = _prep_batch(X_train,Y_train)
    let x_slow = temparr[0];
    let x_mid = temparr[1];
    let x_prev = temparr[2];
    let target_train = temparr[3]; 
    // Each matrix of subsequences for the three modules contains as many overlapped
    // subsequences as necessary to encompass the entire audio clip time series vector
    // under investigation. Due to memory restrictions, it is not possible to fit
    // all of these overlapped subsequences at the same time as one batch. Rather,
    // the desired complete batch will be split up into 13 smaller mini-batches of size
    // 36 and trained sequentially in this manner. 
    let counter_arr = [0,1,2,3,4,5,6,7,8,9,10,11,12];
    let x_slow_split = tf.split(x_slow,13);
    let x_mid_split = tf.split(x_mid,13);
    let x_prev_split = tf.split(x_prev,13);
    let target_train_split = tf.split(target_train,13);
    for (const item of counter_arr) {
        await model.fit([x_slow_split[item], x_mid_split[item], x_prev_split[item]], target_train_split[item],{batchSize: 36,
        epochs: 1});
    }
    tf.dispose(temparr);
    tf.dispose(x_slow);
    tf.dispose(x_mid);
    tf.dispose(x_prev);
    tf.dispose(target_train);
    tf.dispose(x_slow_split);
    tf.dispose(x_mid_split);
    tf.dispose(x_prev_split);
    tf.dispose(target_train_split);
}

/**
 * Function to train the SampleRNN iteratively using train_on_batch() for as many
 * elements as there are in the dataset
 * @param X_train Matrix of the desired subsequences with normalized values.
 * @param Y_train Matrix of the desired subsequences with unnormalized, quantized levels
 */
async function train_iter(X_train,Y_train) {
    for (let ii=0; ii < X_train.length; ii++) {
        // Randomly select an audio clip time series vector from the dataset
        ind = Math.floor(Math.random() * (X_train.length));
        // Split the chosen audio clip time series vector into overlapped subsequences
        // using split_subsequences()
        X_subseq_array = split_subsequences(X_train,ind);
        Y_subseq_array = split_subsequences(Y_train,ind);
        // Train the model on the subsequences of the audio clip time series vector
        // using train_on_batch() 
        await train_on_batch(X_subseq_array,Y_subseq_array,srnn_final);
        // Save the model architecture and weights once training is complete
        await save_model(srnn_final);
        console.log('Finished a batch');
    }
}

/**
 * Function to save the model
 * @param model Trained neural network model
 */
async function save_model(model) {
    await model.save('file://my-model');
}

/**
 * Function to load the SampleRNN
 */
async function load_model() {
    srnn = await tf.loadModel('file://my-model/model.json');
    return srnn;
}

/**
 * Helper function to convert JSON file to 2D array
 * @param json JSON file
 */
function convertJSONto2D(json) {
    var d2Array = [];
    var result  = [];

    for(key in json) {
        result.push(json[key]);
    }
    d2Array.push(result);
    return d2Array[0];
}        

// The SampleRNN was first trained for a few epochs in Python to learn appropriate
// (non-random) initial values for the weights.
// The SampleRNN architecture is manually re-created layer by layer in JS and the 
// weights, saved as JSON files, are loaded into the appropriate layers.
/**
 * Function to create SampleRNN architecture and load in learned weights
 */
function load_models() {
    // Load in the weights for each layer
    var url = 'whale/slow_rnn_pred_0.json';
    slow_pred_rnn_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/slow_rnn_pred_1.json';
    slow_pred_rnn_1 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/slow_rnn_pred_2.json';
    slow_pred_rnn_2 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/slow_td_0.json';
    slow_td_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/slow_td_1.json';
    slow_td_1 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/mid_tdpre_0.json';
    mid_tdpre_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/mid_tdpre_1.json';
    mid_tdpre_1 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/mid_rnn_pred_0.json';
    mid_pred_rnn_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/mid_rnn_pred_1.json';
    mid_pred_rnn_1 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/mid_rnn_pred_2.json';
    mid_pred_rnn_2 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/mid_tdpost_0.json';
    mid_tdpost_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/mid_tdpost_1.json';
    mid_tdpost_1 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/top_embed_0.json';
    top_embed_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/top_tdpre_0.json';
    top_tdpre_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/top_td1_0.json';
    top_td1_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/top_td1_1.json';
    top_td1_1 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/top_td2_0.json';
    top_td2_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/top_td2_1.json';
    top_td2_1 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/top_td3_0.json';
    top_td3_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/top_td3_1.json';
    top_td3_1 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/X_train.json';
    X_train = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/Y_train.json';
    Y_train = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/slow_rnn_initialstate.json';
    slow_rnn_h_0 = JSON.parse(fs.readFileSync(url),'utf8');
    url = 'whale/mid_rnn_initialstate.json';
    mid_rnn_h_0 = JSON.parse(fs.readFileSync(url),'utf8');

    // Convert the JSON files for the initial states of the GRUs into tensors
    let slow_rnn_h_2d = convertJSONto2D(slow_rnn_h_0);
    slow_rnn_h = tf.tensor2d(slow_rnn_h_2d,[1,256]);
    slow_rnn_h0 = slow_rnn_h.tile([batch_size,1])
    let mid_rnn_h_2d = convertJSONto2D(mid_rnn_h_0);
    mid_rnn_h = tf.tensor2d(mid_rnn_h_2d,[1,256]);
    mid_rnn_h0 = mid_rnn_h.tile([batch_size,1])

    // Manually re-create the architecture of the SampleRNN. This is necessary in o
    // order to be able to specify the initial states of the GRUs
    srnn = [];
    srnn[0] = tf.input({shape:[256,1]});
    srnn[1] = tf.layers.reshape({targetShape: [32,8]}).apply(srnn[0]);
    srnn[3] = tf.layers.gru({units:256,returnSequences:true,recurrentActivation:'sigmoid',initialState:slow_rnn_h0}).apply(srnn[1]);
    srnn[5] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 1024})}).apply(srnn[3]);
    srnn[7] = tf.layers.reshape({targetShape: [128,256]}).apply(srnn[5]);

    srnn[2] = tf.input({shape:[256,1]});
    srnn[4] = tf.layers.reshape({targetShape: [128,2]}).apply(srnn[2]);
    srnn[6] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256})}).apply(srnn[4]);
    srnn[8] = tf.layers.add().apply([srnn[6],srnn[7]]);
    srnn[10] = tf.layers.gru({units:256,returnSequences:true,recurrentActivation:'sigmoid',initialState:mid_rnn_h0}).apply(srnn[8]);
    srnn[12] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 512})}).apply(srnn[10]);
    srnn[14] = tf.layers.reshape({targetShape: [256,256]}).apply(srnn[12]);

    srnn[9] = tf.input({shape:[256,1]});
    srnn[11] = tf.layers.reshape({targetShape: [256]}).apply(srnn[9]);
    srnn[13] = tf.layers.embedding({inputDim: 256, outputDim: 256, embeddingsInitializer: tf.initializers.randomNormal({mean: 0, stddev: 1.})}).apply(srnn[11]); 
    srnn[15] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256, useBias: false, kernelInitializer: 'leCunNormal'})}).apply(srnn[13]);
    srnn[16] = tf.layers.add().apply([srnn[14],srnn[15]]);
    srnn[17] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256, activation: 'relu'})}).apply(srnn[16]);
    srnn[18] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256, activation: 'relu'})}).apply(srnn[17]);
    srnn[19] = tf.layers.timeDistributed({layer: tf.layers.dense({units: 256, kernelInitializer: 'leCunNormal'})}).apply(srnn[18]);
    srnn_final = tf.model({inputs: [srnn[0],srnn[2],srnn[9]], outputs: srnn[19]});
    // Compile the SampleRNN using the custom "categorical_crossentropy" loss function
    // and Adam optimizer
    srnn_final.compile({
        optimizer: optimizer,
        loss: categorical_crossentropy,
        metrics: ['accuracy'],
    });

    // Convert the JSON files of the weights into 2D arrays. Then convert the 2D arrays
    // into tensors so that they can be set as the weights for the appropriate layers
    // of the SampleRNN.
    let slow_pred_rnn_0_2d = convertJSONto2D(slow_pred_rnn_0);
    let slow_pred_rnn_1_2d = convertJSONto2D(slow_pred_rnn_1);
    let slow_pred_rnn_2_2d = convertJSONto2D(slow_pred_rnn_2);
    slow_pred_rnn = [tf.tensor2d(slow_pred_rnn_0_2d,[8,768]),tf.tensor2d(slow_pred_rnn_1_2d,[256,768]),tf.tensor1d(slow_pred_rnn_2_2d)];
    srnn_final.layers[3].setWeights(slow_pred_rnn);
    tf.dispose(slow_pred_rnn_0_2d);
    tf.dispose(slow_pred_rnn_1_2d);
    tf.dispose(slow_pred_rnn_2_2d);
    tf.dispose(slow_pred_rnn_0);
    tf.dispose(slow_pred_rnn_1);
    tf.dispose(slow_pred_rnn_2);
    tf.dispose(slow_pred_rnn);
    srnn_final.layers[3]._trainableWeights.push(slow_rnn_h);

    let slow_td_0_2d = convertJSONto2D(slow_td_0);
    let slow_td_1_2d = convertJSONto2D(slow_td_1);
    slow_td = [tf.tensor2d(slow_td_0_2d,[256,1024]),tf.tensor1d(slow_td_1_2d)];
    srnn_final.layers[5].setWeights(slow_td);
    tf.dispose(slow_td_0_2d);
    tf.dispose(slow_td_1_2d);
    tf.dispose(slow_td_0);
    tf.dispose(slow_td_1);
    tf.dispose(slow_td);

    let mid_tdpre_0_2d = convertJSONto2D(mid_tdpre_0);
    let mid_tdpre_1_2d = convertJSONto2D(mid_tdpre_1);
    mid_tdpre = [tf.tensor2d(mid_tdpre_0_2d,[2,256]),tf.tensor1d(mid_tdpre_1_2d)];
    srnn_final.layers[6].setWeights(mid_tdpre);
    tf.dispose(mid_tdpre_0_2d);
    tf.dispose(mid_tdpre_1_2d);
    tf.dispose(mid_tdpre_0);
    tf.dispose(mid_tdpre_1);
    tf.dispose(mid_tdpre);
    let mid_pred_rnn_0_2d = convertJSONto2D(mid_pred_rnn_0);
    let mid_pred_rnn_1_2d = convertJSONto2D(mid_pred_rnn_1);
    let mid_pred_rnn_2_2d = convertJSONto2D(mid_pred_rnn_2);

    mid_pred_rnn = [tf.tensor2d(mid_pred_rnn_0_2d,[256,768]),tf.tensor2d(mid_pred_rnn_1_2d,[256,768]),tf.tensor1d(mid_pred_rnn_2_2d)];
    srnn_final.layers[10].setWeights(mid_pred_rnn);
    tf.dispose(mid_pred_rnn_0_2d);
    tf.dispose(mid_pred_rnn_1_2d);
    tf.dispose(mid_pred_rnn_2_2d);
    tf.dispose(mid_pred_rnn_0);
    tf.dispose(mid_pred_rnn_1);
    tf.dispose(mid_pred_rnn_2);
    tf.dispose(mid_pred_rnn);
    srnn_final.layers[10]._trainableWeights.push(mid_rnn_h);

    let mid_tdpost_0_2d = convertJSONto2D(mid_tdpost_0);
    let mid_tdpost_1_2d = convertJSONto2D(mid_tdpost_1);
    mid_tdpost = [tf.tensor2d(mid_tdpost_0_2d,[256,512]),tf.tensor1d(mid_tdpost_1_2d)];
    srnn_final.layers[12].setWeights(mid_tdpost);
    tf.dispose(mid_tdpost_0_2d);
    tf.dispose(mid_tdpost_1_2d);
    tf.dispose(mid_tdpost_0);
    tf.dispose(mid_tdpost_1);
    tf.dispose(mid_tdpost);

    let top_embed_0_2d = convertJSONto2D(top_embed_0);
    top_embed = [tf.tensor2d(top_embed_0_2d,[256,256])];
    srnn_final.layers[13].setWeights(top_embed);
    tf.dispose(top_embed);
    tf.dispose(top_embed_0_2d);
    tf.dispose(top_embed_0);
    let top_tdpre_0_2d = convertJSONto2D(top_tdpre_0);
    top_tdpre = [tf.tensor2d(top_tdpre_0_2d,[256,256])];
    srnn_final.layers[15].setWeights(top_tdpre);
    tf.dispose(top_tdpre);
    tf.dispose(top_tdpre_0);
    tf.dispose(top_tdpre_0_2d);
    let top_td1_0_2d = convertJSONto2D(top_td1_0);
    let top_td1_1_2d = convertJSONto2D(top_td1_1);
    top_td1 = [tf.tensor2d(top_td1_0_2d,[256,256]),tf.tensor1d(top_td1_1_2d)];
    srnn_final.layers[17].setWeights(top_td1);
    tf.dispose(top_td1_0_2d);
    tf.dispose(top_td1_1_2d);
    tf.dispose(top_td1_0);
    tf.dispose(top_td1_1);
    tf.dispose(top_td1);
    let top_td2_0_2d = convertJSONto2D(top_td2_0);
    let top_td2_1_2d = convertJSONto2D(top_td2_1);
    top_td2 = [tf.tensor2d(top_td2_0_2d,[256,256]),tf.tensor1d(top_td2_1_2d)];
    srnn_final.layers[18].setWeights(top_td2);
    tf.dispose(top_td2_0_2d);
    tf.dispose(top_td2_1_2d);
    tf.dispose(top_td2_0);
    tf.dispose(top_td2_1);
    tf.dispose(top_td2);
    let top_td3_0_2d = convertJSONto2D(top_td3_0);
    let top_td3_1_2d = convertJSONto2D(top_td3_1);
    top_td3 = [tf.tensor2d(top_td3_0_2d,[256,256]),tf.tensor1d(top_td3_1_2d)];
    srnn_final.layers[19].setWeights(top_td3);
    tf.dispose(top_td3_0_2d);
    tf.dispose(top_td3_1_2d);
    tf.dispose(top_td3_0);
    tf.dispose(top_td3_1);
    tf.dispose(top_td3);
    // Convert the "X_train" and "Y_train" dataset JSON files into 2D arrays
    X_train = convertJSONto2D(X_train);
    Y_train = convertJSONto2D(Y_train);
}
// Load in the data and train the SampleRNN model.
load_models(); 
train_iter(X_train,Y_train);
