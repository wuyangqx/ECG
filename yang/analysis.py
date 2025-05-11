import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def plot_wave(records, idx, time_interval = None, title=None):
    record = records[idx]
    dfP = record['df_wave'].copy()
    if time_interval is not None:
        dfP = dfP[(dfP['Time'] >= time_interval[0]) & (dfP['Time'] <= time_interval[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfP['Time'], y=dfP['MLII'], mode='lines', name='MLII'))
    fig.update_layout(title=title + f" sample {idx}", xaxis_title='Time (s)', yaxis_title='mV')
    fig.show()

def plot_accuracy_and_loss(history, X_test, y_test, model, accuracy_keys = ['custom_accuracy', 'val_custom_accuracy'], loss_keys = ['loss', 'val_loss']):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history[accuracy_keys[0]])
    plt.plot(history.history[accuracy_keys[1]])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history[loss_keys[0]])
    plt.plot(history.history[loss_keys[1]])
    plt.show()


import itertools
import numpy as np
def plot_confusion_matrix(cm, classes,
                          normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(6, 6))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()