from typing import Tuple


class CallbackList(object):
    """
    Container abstracting a list of callbacks

    # Arguments
        callbacks: a list of Callback instances
    """
    def __init__(self, callbacks):
        self.callbacks = [c for c in callbacks]

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """
        Called at the end of training

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of an epoch

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """
        Called at the start of a batch

        # Arguments
            batch: integer, index of batch whithin current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch

        # Arguments
            batch: integer, index of batch whithin current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class Callback(object):
    def __init__(self):
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass