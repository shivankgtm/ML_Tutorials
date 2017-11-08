import tensorflow as tf

# Prepare training data
training_inputs = tf.placeholder(shape=[None,3], dtype=tf.float32)
training_outputs = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Preparing Neural Networks parameteres

weights = tf.Variable(initial_value=[[.3], [.1], [.8]], dtype=tf.float32)
bias = tf.Variable(initial_value=[[1]], dtype = tf.float32)


# Preparing inputs of activation function

af_input = tf.matmul(training_inputs, weights) + bias


# activate function of output layer neuron

predictions = tf.nn.sigmoid(af_input)

# Measuring the prediction error of the network after being Trained

prediction_error = tf.reduce_sum(training_outputs - predictions)


# Minimizing The predictions error
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(prediction_error)

# creating a tensorflow session

sees = tf.Session()

# Inititializing the tensorflow variables  W and b
sees.run(tf.global_variables_initializer())

training_inputs_data = [[1,1,1],
                       [0,0,0],
                       [1,0,1],
                       [0,1,1]]

training_outputs_data = [[1],[0],[0],[0]]

# training loop of neural network

for step in range(10000):
    sees.run(fetches=[train_op], feed_dict={
        training_inputs: training_inputs_data,
                                      training_outputs: training_outputs_data
    })

# classes scores of some testing data

print('Expected scores : ', sees.run(fetches=predictions, feed_dict={training_inputs: [[0, 0, 0],[1,1,1],[0,1,0],[0,0,1]]}))

sees.close()