import tensorflow as tf
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
import random
from numba import prange
import numpy as np

clients = 100 #100
rounds = 50 #50
clients_per_round = 15 #15
local_epochs = 10 #10
beta = .7

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

def model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

model_of_client = []
x_of_client = []
y_of_client = []
samples_per_client = int(len(x_train)/clients)
for i in range(clients):
    model_of_client.append(model())
    x_of_client.append(x_train[i*samples_per_client:(i+1)*samples_per_client])
    y_of_client.append(y_train[i*samples_per_client:(i+1)*samples_per_client])

global_model = model().get_weights()[:4]
# update global model in each client network
for m in model_of_client:
    w = m.get_weights()
    w[:4] = global_model
    m.set_weights(w)
round_acc = []
round_loss = []
for round in range(rounds):
    print("Round " + str(round+1))
    
    round_deltas = []
    round_client_ids = random.sample(range(clients), clients_per_round)
    for client_id_id in prange(clients_per_round):
        round_deltas.append([np.zeros(x.shape) for x in global_model])    
        client_id = round_client_ids[client_id_id]
        client_model = model_of_client[client_id]
        x = x_of_client[client_id]
        y = y_of_client[client_id]
        old_w = client_model.get_weights()[:4]
        client_model.fit(x, y, batch_size=60, epochs=local_epochs, verbose=0)
        round_deltas[client_id_id] = [round_deltas[client_id_id][i] + client_model.get_weights()[:4][i] - old_w[i] for i in range(4)]
        print(".", end="", flush=True)
    
    print()
    round_delta_sum = [np.zeros(x.shape) for x in global_model]
    for i in range(0, clients_per_round):
        for j in range(4):
            round_delta_sum[j] += round_deltas[i][j]
    round_delta_sum = [round_delta_sum[j] / clients_per_round for j in range(4)]
    global_model = [global_model[j] + round_delta_sum[j] * beta for j in range(4)]
    
    # update global model in each client network
    for m in model_of_client:
        w = m.get_weights()
        w[:4] = global_model
        m.set_weights(w)

    # calculate average loss and accuracy
    total_loss = np.zeros((clients))
    total_acc = np.zeros((clients))
    for i in prange(clients):
        m = model_of_client[i]
        loss, acc = m.evaluate(x_test[:100],  y_test[:100], verbose=0)
        total_acc[i] = acc
        total_loss[i] = loss
        print(".", end="", flush=True)
    print()
    avg_loss = np.sum(total_loss) / clients
    avg_acc = np.sum(total_acc) / clients
    print("Loss: "+str(avg_loss))
    print("Accuracy: "+str(avg_acc))
    round_acc.append(avg_acc)
    round_loss.append(avg_loss)

plt.plot(round_loss)
plt.ylabel('Loss')
plt.xlabel('Round')
plt.savefig('loss_over_rounds_plot.png')

plt.clf()
plt.plot(round_acc)
plt.ylabel('Accuracy')
plt.xlabel('Round')
plt.savefig('accuracy_over_rounds_plot.png')

for i, m in enumerate(model_of_client):
    print("Client "+str(i+1))
    test_loss, test_acc = m.evaluate(x_test,  y_test, verbose=2)
    print(test_loss)
    print(test_acc)
