import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel

class GCNModel(TorchModel):
  """Model for Molecular Property Prediction Based on Thomas N. Kipf et al.

  This model has the following features:

  * For each graph convolution, the learnable weight in this model is shared across all nodes.
    ``GraphConvModel`` employs separate learnable weights for nodes of different degrees. A
    learnable weight is shared across all nodes of a particular degree.
  * For ``GraphConvModel``, there is an additional GraphPool operation after each
    graph convolution. The operation updates the representation of a node by applying an
    element-wise maximum over the representations of its neighbors and itself.
  * For computing graph-level representations, this model computes a weighted sum and an
    element-wise maximum of the representations of all nodes in a graph and concatenates them.
    The node weights are obtained by using a linear/dense layer followd by a sigmoid function.
    For ``GraphConvModel``, the sum over node representations is unweighted.
  * There are various minor differences in using dropout, skip connection and batch
    normalization.
  """

  def __init__(self,
               n_tasks: int,
               graph_conv_layers: list = None,
               activation=None,
               residual: bool = True,
               batchnorm: bool = False,
               dropout: float = 0.2,
               predictor_hidden_feats: int = 128,
               predictor_dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features=30,
               n_classes: int = 2,
               nfeat_name: str = 'x',
               self_loop: bool = True,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    graph_conv_layers: list of int
      Width of channels for GCN layers. graph_conv_layers[i] gives the width of channel
      for the i-th GCN layer. If not specified, the default value will be [64, 64].
    activation: callable
      The activation function to apply to the output of each GCN layer.
      By default, no activation function will be applied.
    residual: bool
      Whether to add a residual connection within each GCN layer. Default to True.
    batchnorm: bool
      Whether to apply batch normalization to the output of each GCN layer.
      Default to False.
    dropout: float
      The dropout probability for the output of each GCN layer. Default to 0.
    predictor_hidden_feats: int
      The size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout: float
      The dropout probability in the output MLP predictor. Default to 0.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    number_atom_features: int
      The length of the initial atom feature vectors. Default to 30.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    nfeat_name: str
      For an input graph ``g``, the model assumes that it stores node features in
      ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
      Default to 'x'.
    self_loop: bool
      Whether to add self loops for the nodes, i.e. edges from nodes to themselves.
      Default to True.
    kwargs
      This can include any keyword argument of TorchModel.
    """
    model = GCN(
        n_tasks=n_tasks,
        graph_conv_layers=graph_conv_layers,
        activation=activation,
        residual=residual,
        batchnorm=batchnorm,
        dropout=dropout,
        predictor_hidden_feats=predictor_hidden_feats,
        predictor_dropout=predictor_dropout,
        mode=mode,
        number_atom_features=number_atom_features,
        n_classes=n_classes,
        nfeat_name=nfeat_name)
    if mode == 'regression':
      loss: Loss = L2Loss()
      output_types = ['prediction']
    else:
      loss = SparseSoftmaxCrossEntropy()
      output_types = ['prediction', 'loss']
    super(GCNModel, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

    self._self_loop = self_loop

  def _prepare_batch(self, batch):
    """Create batch data for GCN.

    Parameters
    ----------
    batch: tuple
      The tuple is ``(inputs, labels, weights)``.
    self_loop: bool
      Whether to add self loops for the nodes, i.e. edges from nodes
      to themselves. Default to False.

    Returns
    -------
    inputs: DGLGraph
      DGLGraph for a batch of graphs.
    labels: list of torch.Tensor or None
      The graph labels.
    weights: list of torch.Tensor or None
      The weights for each sample or sample/task pair converted to torch.Tensor.
    """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')

    inputs, labels, weights = batch
    dgl_graphs = [
        graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
    ]
    inputs = dgl.batch(dgl_graphs).to(self.device)
    _, labels, weights = super(GCNModel, self)._prepare_batch(([], labels,
                                                               weights))
    return inputs, labels, weights
