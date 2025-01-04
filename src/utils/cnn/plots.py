from matplotlib import pyplot as plt

class training_metrics:

    def __init__(self, history):
        self.history = history

    def plot_loss(self):
        """
        Plot the training and validation loss over epochs.

        Returns:
            matplotlib.figure.Figure: Figure containing the loss plot.
        """
        fig, ax = plt.subplots()
        ax.plot(self.history.history['loss'], color='teal', label='loss')
        ax.plot(self.history.history['val_loss'], color='orange', label='val_loss')
        ax.set_title('Loss', fontsize=20)
        ax.legend(loc="upper left")
        return fig

    def plot_accuracy(self):
        """
        Plot the training and validation accuracy over epochs.

        Returns:
            matplotlib.figure.Figure: Figure containing the accuracy plot.
        """
        fig, ax = plt.subplots()
        ax.plot(self.history.history['accuracy'], color='teal', label='accuracy')
        ax.plot(self.history.history['val_accuracy'], color='orange', label='val_accuracy')
        ax.set_title('Accuracy', fontsize=20)
        ax.legend(loc="upper left")
        return fig