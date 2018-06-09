import numpy as np

import backend
import nn


class Model(object):
    """Base model class for the different applications"""

    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)


class RegressionModel(Model):
    """
    A neural network model for app roximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 1
        self.graph = None

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.
        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.
        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values
        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            size = len(x)//4                         
            w1 = nn.Variable(size,size)
            w2 = nn.Variable(size,size)
            w3 = nn.Variable(size,size)
            w4 = nn.Variable(size,size)
            b1 = nn.Variable(size,1)
            b2 = nn.Variable(size,1)
            b3 = nn.Variable(size,1)
            b4 = nn.Variable(size,1)

            self.graph = nn.Graph([w1,w2,w3,w4,b1,b2,b3,b4])

            input_x = nn.Input(self.graph,x)
            input_y = nn.Input(self.graph,y)

            x1 = nn.Input(self.graph,x[:size])
            x2 = nn.Input(self.graph,x[size:len(x)//2])
            x3 = nn.Input(self.graph,x[len(x)//2:3*size])
            x4 = nn.Input(self.graph,x[3*size:])

            mult1 = nn.MatrixMultiply(self.graph, w1, x1)
            mult2 = nn.MatrixMultiply(self.graph, w2, x2)
            mult3 = nn.MatrixMultiply(self.graph, w3, x3)
            mult4 = nn.MatrixMultiply(self.graph, w4, x4)

            add1 = nn.MatrixVectorAdd(self.graph, mult1, mult2)
            add2 = nn.MatrixVectorAdd(self.graph, mult2, mult1)

            add3 = nn.MatrixVectorAdd(self.graph, mult3, mult4)
            add4 = nn.MatrixVectorAdd(self.graph, mult4, mult3)
            add5 = nn.MatrixVectorAdd(self.graph, add3, b3)
            add6 = nn.MatrixVectorAdd(self.graph, add4, b4)

            y1 = nn.Input(self.graph,y[:len(y)//4])
            y2 = nn.Input(self.graph,y[len(y)//4:len(y)//2])
            y3 = nn.Input(self.graph,y[len(y)//2:3*len(y)//4])
            y4 = nn.Input(self.graph,y[3*len(y)//4:])

            loss1 = nn.SquareLoss(self.graph, add5, y1)
            loss2 = nn.SquareLoss(self.graph, add6, y2)
            loss3 = nn.SquareLoss(self.graph, add6, y3)
            loss4 = nn.SquareLoss(self.graph, add6, y4)

            nn.Add(self.graph, nn.Add(self.graph, nn.Add(self.graph, loss1, loss2), loss3), loss4)
            
            return self.graph

        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            return np.concatenate((np.concatenate((np.concatenate((self.graph.get_output(self.graph.get_nodes()[-11]), self.graph.get_output(self.graph.get_nodes()[-10])), axis=0), self.graph.get_output(self.graph.get_nodes()[-9])), axis=0), self.graph.get_output(self.graph.get_nodes()[-8])), axis=0)


class OddRegressionModel(Model):
    """
    TODO: Question 5 - [Application] OddRegression

    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.graph = None
        self.list = []

    def run(self, x, y=None):
        """
        TODO: Question 5 - [Application] OddRegression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        n = 4
        if not self.graph:
                self.list = [nn.Variable(1, 50),nn.Variable(50, 50),nn.Variable(50, 1),nn.Variable(1, 50),nn.Variable(1, 50),nn.Variable(1, 1)]
        self.graph = nn.Graph(self.list)
        input_x = nn.Input(self.graph,x)
        input_neg = nn.Input(self.graph, np.matrix([-1.]))

        ad = nn.MatrixVectorAdd(
            self.graph, nn.MatrixMultiply(
                self.graph, nn.ReLU(
                    self.graph, nn.MatrixVectorAdd(
                        self.graph, nn.MatrixMultiply(
                            self.graph, nn.ReLU(
                                self.graph, nn.MatrixVectorAdd(
                                    self.graph, nn.MatrixMultiply(
                                        self.graph, input_x, self.list[0]), self.list[3])), self.list[1]), self.list[4])), self.list[2]), self.list[5])

        sb = nn.MatrixMultiply(
            self.graph, nn.MatrixVectorAdd(
                self.graph, nn.MatrixMultiply(
                    self.graph, nn.ReLU(
                        self.graph, nn.MatrixVectorAdd(
                            self.graph, nn.MatrixMultiply(
                                self.graph, nn.ReLU(
                                    self.graph, nn.MatrixVectorAdd(
                                        self.graph, nn.MatrixMultiply(
                                            self.graph, nn.MatrixMultiply(
                                                self.graph, input_x, input_neg), self.list[0]), self.list[3])), self.list[1]), self.list[4])), self.list[2]), self.list[5]), input_neg)

        # f(x) = g(x)-g(-x)
        sub = nn.MatrixVectorAdd(self.graph, ad, sb)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.graph,y)
            loss = nn.SquareLoss(self.graph, sub, input_y)
            return self.graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(self.graph.get_nodes()[-1])


class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.
    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.
    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.2
        self.graph = None

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.
        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.
        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.
        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"

        if len(x) == 1:
            return 0

        if not self.graph:
                self.list = [nn.Variable(784, 500),nn.Variable(500, 500),nn.Variable(500, 10),nn.Variable(1, 500),nn.Variable(1, 500),nn.Variable(1, 10)]

        self.graph = nn.Graph(self.list)
        input_x = nn.Input(self.graph,x)

        temp = nn.MatrixVectorAdd(
            self.graph, nn.MatrixMultiply(
                self.graph, nn.ReLU(
                    self.graph, nn.MatrixVectorAdd(
                        self.graph, nn.MatrixMultiply(
                            self.graph, nn.ReLU(
                                self.graph, nn.MatrixVectorAdd(
                                    self.graph, nn.MatrixMultiply(
                                        self.graph, input_x, self.list[0]), self.list[3])), self.list[1]), self.list[4])), self.list[2]), self.list[5])
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            input_y = nn.Input(self.graph,y)
            loss = nn.SoftmaxLoss(self.graph, temp, input_y)
            return self.graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            return self.graph.get_output(self.graph.get_nodes()[-1])

class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.graph = None

    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.
        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.
        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"

        if not self.graph:
            self.list = [nn.Variable(4, 50),nn.Variable(50, 50),nn.Variable(50, 2),nn.Variable(1, 50),nn.Variable(1, 50),nn.Variable(1, 2)]

        self.graph = nn.Graph(self.list)
        input_x = nn.Input(self.graph,states)

        temp = nn.MatrixVectorAdd(
            self.graph, nn.MatrixMultiply(
                self.graph, nn.ReLU(
                    self.graph, nn.MatrixVectorAdd(
                        self.graph, nn.MatrixMultiply(
                            self.graph, nn.ReLU(
                                self.graph, nn.MatrixVectorAdd(
                                    self.graph, nn.MatrixMultiply(
                                        self.graph, input_x, self.list[0]), self.list[3])), self.list[1]), self.list[4])), self.list[2]), self.list[5])

        if Q_target is not None:
            "* YOUR CODE HERE *"
            input_y = nn.Input(self.graph, Q_target)
            loss = nn.SquareLoss(self.graph, temp, input_y)
            return self.graph
        else:
            "* YOUR CODE HERE *"
            return self.graph.get_output(self.graph.get_nodes()[-1])

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.
        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.
    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # Y can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        #0.032 79.8%
        #0.034 83.8%
        #0.045 79.4%
        self.learning_rate = 0.034
        self.graph = None

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.
        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).
        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.
        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.
        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)
        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"

        if not self.graph:
            self.l = [nn.Variable(self.num_chars, self.num_chars),nn.Variable(self.num_chars, self.num_chars),nn.Variable(1, self.num_chars),nn.Variable(1, self.num_chars),nn.Variable(1, self.num_chars),nn.Variable(self.num_chars, self.num_chars),nn.Variable(self.num_chars, 80),nn.Variable(1, self.num_chars),nn.Variable(1, 80), nn.Variable(self.num_chars, self.num_chars), nn.Variable(1, self.num_chars),nn.Variable(80, 5),nn.Variable(1, 5)]

        self.graph = nn.Graph(self.l)

        "* YOUR CODE HERE *"
        char_inputs = [] 
        temp = nn.MatrixVectorAdd(
            self.graph, nn.Input(
                self.graph,np.zeros(
                    (batch_size, self.num_chars))), self.l[4])

        for i in range(len(xs)):
            char_inputs.append(nn.Input(self.graph, xs[i])) 
            temp = nn.ReLU(
                self.graph, nn.MatrixVectorAdd(
                    self.graph, nn.MatrixMultiply(
                        self.graph, nn.MatrixVectorAdd(
                            self.graph, temp, char_inputs[i]), self.l[0]), self.l[2]) )

        add3 = nn.MatrixVectorAdd(
            self.graph, nn.MatrixMultiply(
                self.graph, nn.ReLU(
                    self.graph, nn.MatrixVectorAdd(
                        self.graph, nn.MatrixMultiply(
                            self.graph, temp, self.l[6]), self.l[8])), self.l[11]), self.l[12])

        if y is not None:
            "* YOUR CODE HERE *"
            input_y = nn.Input(self.graph, y)
            loss = nn.SoftmaxLoss(self.graph, add3, input_y)
            return self.graph
        else:
            "* YOUR CODE HERE *"
            return self.graph.get_output(self.graph.get_nodes()[-1])
