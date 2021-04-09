import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel


class GCN(nn.Module):
  """Model based on Graph Convolution Networks (GCN) by Thomas N. Kipf et al.
  
  This model is different from deepchem.models.GraphConvModel as follows:
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
               dropout: float = 0.,
               predictor_hidden_feats: int = 128,
               predictor_dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features: int = 30,
               n_classes: int = 2,
               nfeat_name: str = 'x'):
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
    """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')
    try:
      import dgllife
    except:
      raise ImportError('This class requires dgllife.')

    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    super(GCN, self).__init__()

    self.n_tasks = n_tasks
    self.mode = mode
    self.n_classes = n_classes
    self.nfeat_name = nfeat_name
    if mode == 'classification':
      out_size = n_tasks * n_classes
    else:
      out_size = n_tasks

    from dgllife.model import GCNPredictor as DGLGCNPredictor

    if graph_conv_layers is None:
      graph_conv_layers = [64, 64]
    num_gnn_layers = len(graph_conv_layers)

    if activation is not None:
      activation = [activation] * num_gnn_layers

    self.model = DGLGCNPredictor(
        in_feats=number_atom_features,
        hidden_feats=graph_conv_layers,
        activation=activation,
        residual=[residual] * num_gnn_layers,
        batchnorm=[batchnorm] * num_gnn_layers,
        dropout=[dropout] * num_gnn_layers,
        n_tasks=out_size,
        predictor_hidden_feats=predictor_hidden_feats,
        predictor_dropout=predictor_dropout)

  def forward(self, g):
    """Predict graph labels
    Parameters
    ----------
    g: DGLGraph
      A DGLGraph for a batch of graphs. It stores the node features in
      ``dgl_graph.ndata[self.nfeat_name]``.
    Returns
    -------
    torch.Tensor
      The model output.
      * When self.mode = 'regression',
        its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
      * When self.mode = 'classification', the output consists of probabilities
        for classes. Its shape will be ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)``
        if self.n_tasks > 1; its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if
        self.n_tasks is 1.
      torch.Tensor, optional
        This is only returned when self.mode = 'classification', the output consists of the
        logits for classes before softmax.
    """
    node_feats = g.ndata[self.nfeat_name]
    out = self.model(g, node_feats)

    if self.mode == 'classification':
      if self.n_tasks == 1:
        logits = out.view(-1, self.n_classes)
        softmax_dim = 1
      else:
        logits = out.view(-1, self.n_tasks, self.n_classes)
        softmax_dim = 2
      proba = F.softmax(logits, dim=softmax_dim)
      return proba, logits
    else:
      return out


class GCNModel(TorchModel):
  """Model for Molecular Property Prediction Based on GCN.

  This model is an optimised version of the GCN which is used in our experiments.
  The optimisation is in the hyperparameter values which are changed so as to get better regularisation.

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
