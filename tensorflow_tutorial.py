import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


'''
input > weight > hidden_1 (act function) > weights > hidden_2 (acti function) > weights > output

cost function 

optimization function
'''


mnist = input_data.read_data_sets("/tmp/data", one_hot=True )

n_nodes_hl_1 = 500
n_nodes_hl_2 = 500
n_nodes_hl_3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    #  Set up the computation graph
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl_1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl_1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl_1, n_nodes_hl_2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl_2]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl_2, n_nodes_hl_3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl_3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl_3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output

def train_model(x):
    prediction = neural_network_model(x)  # run for each item
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))  # cost function
    optimiser = tf.train.AdamOptimizer().minimize(cost)  # now minimise the cost
    n_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables()) # runs and defines network graph

        for epoch in range(n_epochs):
            epoch_cost = 0
            for _ in range(int(mnist.train.num_examples/batch_size)): # how oft
                           epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                           _, c = sess.run([optimiser, cost], feed_dict = {x: epoch_x, y: epoch_y})
                           epoch_cost += c
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_cost)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_model(x)



