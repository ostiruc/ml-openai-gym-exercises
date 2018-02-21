import tensorflow as tf

class Model:
    def __init__(self, state_size, action_size):
        self._learning_rate = 0.001
        self._x, self._y_hat, self._predict_op, self._train_op = self._build_model(state_size, action_size)

        self._session = tf.InteractiveSession()
        self._session.run(tf.global_variables_initializer())

    def _build_model(self, state_size, action_size):
        # Params
        x = tf.placeholder(tf.float32, [None, state_size])
        y_hat = tf.placeholder(tf.float32, [None, action_size])

        # Layers
        h = self._build_hidden_layer(x, state_size, 24)
        h = self._build_hidden_layer(h, 24, 24)
        y = self._build_output_layer(h, 24, action_size)

        # Operations

        predict_op = tf.nn.softmax(y)

        loss = tf.reduce_mean(
	        tf.nn.softmax_cross_entropy_with_logits(labels=y_hat, logits=y))
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        return x, y_hat, predict_op, train_op

    def _build_hidden_layer(self, input, input_size, output_size):
        W = tf.Variable(
            tf.truncated_normal([input_size, output_size], stddev=0.05))
        
        b = tf.Variable(tf.zeros([output_size]))

        h = tf.nn.relu(tf.matmul(input, W) + b)

        return h

    def _build_output_layer(self, input, input_size, output_size):
        W = tf.Variable(
            tf.truncated_normal([input_size, output_size], stddev=0.05))
        
        b = tf.Variable(tf.zeros([output_size]))

        y = tf.matmul(input, W) + b

        return y

    def predict(self, states):
        y = self._session.run(self._predict_op,
            feed_dict={
                self._x: states})

        return y

    def fit(self, states, actions, epochs, verbose):
        # TODO: Remove the references to epochs and verbose once we've confirmed this all works
        self._session.run(self._train_op, 
            feed_dict={
                self._x: states, 
                self._y_hat: actions})