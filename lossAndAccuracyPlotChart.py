# B00092951 - Matthew Reilly
# Technological University Dublin 2019
# Code to display Loss & Accuracy Plot Chart

#import library
import matplotlib.pyplot as plt

#Functions definition
#Plot Keras Track History
#Plot loss and accuracy for the training and validation set.
def lossAccuracyPlotChartFunction(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Display Loss Plot Chart 
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        
    plt.title('Loss of Training and Validation Chart')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss Measurement')
    plt.legend()
    
    
    
    ## Display Accuracy Plot Chart
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        
    
    plt.title('Accuracy of Training and Validation Chart')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy Measurement')
    plt.legend()
    plt.show()