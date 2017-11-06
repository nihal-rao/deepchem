"""
Convenience class for building sequential deep networks.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
import tensorflow as tf
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import Label
from deepchem.models.tensorgraph.layers import SoftMaxCrossEntropy
from deepchem.models.tensorgraph.layers import ReduceMean


class Sequential(TensorGraph):
  """Sequential models are linear stacks of layers.

  Analogous to the Sequential model from Keras.
  """

  def __init__(self, **kwargs):
    """Initializes a sequential model
    """
    self.num_layers = 0
    self._prev_layer = None
    if "use_queue" in kwargs:
      if kwargs["use_queue"]:
        raise ValueError("Sequential doesn't support queues.")
    kwargs["use_queue"] = False
    self._layer_list = []
    self._built = False
    super(Sequential, self).__init__(**kwargs)

  def add(self, layer):
    """Adds a new layer to model.

    Parameter
    ---------
    layer: Layer
      Adds layer to this graph.
    """
    self._layer_list.append(layer)

  def fit(self, dataset, loss, **kwargs):
    """Fits on the specified dataset.

    Adds the necessary feature and placeholders.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset with data
    loss: string
      Only "binary_crossentropy" for now.
    """
    X_shape, y_shape, _, _ = dataset.get_shape()
    # Calling fit() for first time
    if not self._built:
      feature_shape = X_shape[1:]
      label_shape = y_shape[1:]
      # Add in features
      features = Feature(shape=(None,) + feature_shape)
      self._add_layer(features)
      # Add in labels
      labels = Label(shape=(None,) + label_shape)
      self._add_layer(labels)

      # Add in all layers
      prev_layer = features
      for ind, layer in enumerate(self._layer_list):
        if not len(layer.in_layers) == 0:
          raise ValueError("Cannot specify in_layers for Sequential.")
        layer.in_layers += [prev_layer]
        self._add_layer(layer)
        prev_layer = layer
      # The last layer is the output of the model
      self.outputs.append(prev_layer)

      if loss == "binary_crossentropy":
        smce = SoftMaxCrossEntropy(in_layers=[labels, prev_layer])
        self._add_layer(smce)
        self.set_loss(ReduceMean(in_layers=[smce]))
      else:
        # TODO(rbharath): Add in support for additional losses.
        raise ValueError("Unsupported loss.")
    self._built = True

    super(Sequential, self).fit(dataset, **kwargs)
