import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import sqrt, ceil
import os
import tensorflow as tf
# Force CPU, comment to use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Uncomment to use GPU
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D,Dropout
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score

# Loading pickle files list
pickle_file_list = [ele for ele in os.listdir() if ele.endswith('pickle') and ele.startswith('data')]
# Creating a grid
def image_grid(x_input):
    N, H, W, C = x_input.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + 1 * (grid_size - 1)
    grid_width = W * grid_size + 1 * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C)) + 255
    indeks = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if indeks < N:
                img = x_input[indeks]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)
                indeks += 1
            x0 += W + 1
            x1 += W + 1
        y0 += H + 1
        y1 += H + 1

    return grid


Shd = ['dflt', 'tst']

def scheduler(epoka, lr):
    if epoka < 10:
        print('lr = ',lr)
        return lr
    else:
        print('lr = ',lr * tf.math.exp(-0.1))
        return lr * tf.math.exp(-0.1)
    
def shedulerTest(epoka, lr):
    if epoka != 0:
        decay = lr / epoka
        lr = lr * 1/(1 + decay * epoka)
        print('lr = ',lr)
        return lr
    else:
        print('lr = ',lr)
        return lr
    
# Creating the model
def model_create(epochs, val, modelName = 'myModel', batches = 64, shedulerFunc = 0):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.75))
    model.add(Dense(43, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if shedulerFunc == 0:
        annealer = LearningRateScheduler(scheduler)
    elif shedulerFunc == 1:
        annealer = LearningRateScheduler(shedulerTest)
    else:
        annealer = LearningRateScheduler(scheduler)
    
    h = model.fit(data['x_train'][:val], data['y_train'][:val],
              batch_size = batches, epochs = epochs,
              validation_data = (data['x_validation'], data['y_validation']),
              callbacks = [annealer], verbose=1)
    model.save(modelName+'.h5')
    return h

# Graphing function
def graphs(modele, matrix, wyniki = None, name = "", sizeX = 20.0, sizeY = 15.0):
    columns = list(range(0,len(wyniki)))
    for i in range(len(columns)):
        columns[i] = 'Score nr ' + str(i+1)
    zipped = list(zip(columns, wyniki))
    first_elems = [tupla[0] for tupla in zipped]
    second_elems = [round(tupla[1],4) for tupla in zipped]
    plt.rcParams['figure.figsize'] = (sizeX, sizeY)
    plt.suptitle(name)
    plt.subplot(2,2,1)
    print(matrix)
    for i in range(len(modele)):
        plt.plot(modele[i].history['accuracy'], '-o', linewidth=3.0, label = '{0}e {1}d {2}b {3} s'.format(matrix[0][i], matrix[1][i], matrix[3][i], Shd[S[i]]))
        
    plt.legend(loc = 7)
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20)
    plt.tick_params(labelsize = 18)
        
    plt.subplot(2,2,2)
    for i in range(len(modele)):
        plt.plot(modele[i].history['val_accuracy'], '-o', linewidth=3.0, label = '{0}e {1}d {2}b {3} s'.format(matrix[0][i], matrix[1][i], matrix[3][i], Shd[S[i]]))
    plt.legend(loc = 7)
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel('Validation accuracy', fontsize = 20)
    plt.tick_params(labelsize = 18)
    
    plt.subplot(2,2,3)
    for i in range(len(modele)):
        plt.plot(modele[i].history['val_loss'], '-o', linewidth=3.0, label = '{0}e {1}d {2}b {3} s'.format(matrix[0][i], matrix[1][i], matrix[3][i], Shd[S[i]]))
    plt.legend(loc = 7)
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel('Validation loss', fontsize = 20)
    plt.tick_params(labelsize = 18)
    
    plt.subplot(2,2,4)
    for i in range(len(wyniki)):
        data=[second_elems]             
        plt.table(cellText=data,loc="center",cellLoc='center',colLabels=first_elems).auto_set_column_width(len(first_elems))
    plt.axis('off')
    plt.axis('tight')
    plt.show()
    
# Matrix creator
def matrix_creator(E, D, B = 64, S = 0):
    M = np.zeros((5,len(E)))
    for i in range(len(E)):
        M[0][i] = E[i]
        M[1][i] = D[i]
        M[2][i] = 3
        M[3][i] = B[i]
        M[4][i] = S[i]
    print(M)
    return M.astype(int)
        
# Configuration

E = [50,50,50]
D = [86000,86000,86000]
B = [64,128,256]
S = [0,0,0]
iterator = 0
key_word = ''
score = []
# Model testing
for data in pickle_file_list:
    matrixModel = matrix_creator(E, D, B, S)
    model_list = []
    with open(data, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    # Data loading
    x_train_float = data['x_train'].astype(np.float32)
    data['x_train'] = x_train_float
    y_test_float = data['y_test'].astype(np.float32)
    data['y_test'] = y_test_float
    data['y_train'] = to_categorical(data['y_train'], num_classes=43)
    data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)
    
    # Transposing the color channel
    data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
    data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
    data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)
    
    # Show data
    for i, j in data.items():
        if i == 'labels':
            print(i + ':', len(j))
        else: 
            print(i + ':', j.shape)     
    
    # Show examples
    examples = data['x_train'][:81, :, :, :]
    print(examples.shape)  # (81, 32, 32, 3) 
    
    # Show grid
    fig = plt.figure()
    grid = image_grid(examples)
    plt.imshow(grid.astype('uint8'), cmap='gray')
    plt.axis('off')
    plt.gcf().set_size_inches(15, 15)
    plt.title('Some examples', fontsize=18)
    plt.show()
    plt.close()
    X_test = np.array(data['x_test'])
    for i in range(len(E)):
        print ('{} epok, {} danych uczących, {} rozmiar partii, {} sheduler'.format(E[i], D[i], B[i], Shd[S[i]]))
        model_temp = model_create(E[i], D[i], '{}_epoki_{}_ilosc_danych_{}_partia_{}_sheduler_{}'.format(pickle_file_list[iterator], E[i], D[i], B[i], Shd[S[i]]), B[i], S[i])
        model_list.append(model_temp)
        model = load_model('{}_epoki_{}_ilosc_danych_{}_partia_{}_sheduler_{}'.format(pickle_file_list[iterator], E[i], D[i], B[i], Shd[S[i]]) + '.h5')
        pred = np.argmax(model.predict(X_test), axis=-1)
        score.append(accuracy_score(y_test_float, pred))
        print('<==============================================>')
        print('Score: ' + str(accuracy_score(y_test_float, pred)))
        print('<==============================================>')
        
    graphs(model_list, matrixModel, score, '{} model'.format(pickle_file_list[iterator]) + '\nMaximum score: ' + str(round(max(score),4)) +'\nIndex: ' + str(score.index(max(score))))
    iterator += 1
    score =[]
    if key_word == '11':
        continue
    
    while(key_word != '1' and key_word != '0' and key_word != '11'):
        key_word = input('Go ahead? [1], Stop? [0], Go through all? [11]: \n')
        
    if key_word == '0':
        print(15*'><')
        print('E N D')
        print(15*'><')
        break
    
    