from QuadraticMLP import *
from MultiLayerPerceptron import MLP
import numpy as np
from os.path  import join
from MNIST_DataLoader import MnistDataloader

NN_Test = MLP([784, 128, 64, 10])


input_path = ''
training_images_filepath = join(input_path, 'MNIST_HandwrittenDigits/train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'MNIST_HandwrittenDigits/train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 'MNIST_HandwrittenDigits/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 'MNIST_HandwrittenDigits/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

train_and_save = False
train,save = (1,0)

data_set = mnist_dataloader.load_data(load_train = train or train_and_save,augment=False, replace_set=True, num_of_copies=1)

(x_train, y_train), (x_test, y_test) = data_set

print((len(x_train),len(y_train),len(x_test),len(y_test)))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

sigmoid = lambda x : 1/(1+np.exp(-x))
sigmoid_derivative =lambda x: sigmoid(x)*(1-sigmoid(x))

load = 0
epochs = 1
randomize = False
if train or train_and_save:
    for epoch in range(epochs):
        full_train_size = len(x_train)
        num = full_train_size
        x_train = x_train[:num+1]
        y_train = y_train[:num+1]

        if randomize:
            indices = np.random.permutation(num)
            x_train = x_train[indices]
            y_train = y_train[indices]

        batch_size = 100  # Adjust batch size as needed
        num_batches = num // batch_size

        total_loss = 0.0

        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size

            batch_inputs = np.asarray([np.asarray(x_train[i]).flatten() for i in range(batch_start, batch_end)])
            batch_outputs = np.zeros((batch_size, 10))
            for i in range(batch_size):
                batch_outputs[i][y_train[batch_start + i]] = 1

            total_loss += NN_Test.batchwise_learn(batch_inputs,batch_outputs,learning_rate=batch_size/num,log=False)

            if (batch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch + 1}/{num_batches}, Avg Loss: {total_loss/(batch+1):.8f}")

        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {total_loss / num_batches:.8f}")

    if save or train_and_save:
        NN_Test.save_parameters_text('digit_recog.txt')
if load:
    NN_Test.load_parameters_text('digit_recog.txt')

score = 0
incorrect_answers = []
distribution = [0]*10
actual_distribution = [0]*10
counter = 0

test_set = x_test
test_label = y_test
for i in range(len(test_set)):
    NN_Test.set_inputs(np.asarray(test_set[i]).flatten(),log=False)
    max = np.argmax(NN_Test.get_output())
    distribution[max] += 1
    actual_distribution[test_label[i]] += 1
    if max == test_label[i]:
        score+=1
    else:
        incorrect_answers.append((NN_Test.get_output()[test_label[i]],max,test_label[i]))

ratio = [0]*len(actual_distribution)

for i in range(len(distribution)):
    ratio[i] = round(int(distribution[i])/actual_distribution[i],2)

print(f"Accuracy : {round(100*score/len(test_set),2)}%")
print("Distribution : " + str(distribution))
print("Ratio : "+str(ratio))
