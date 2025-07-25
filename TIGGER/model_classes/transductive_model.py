import torch
import torch.nn as nn
import torch.distributions as D

from .normal import Normal
from .mixture_same_family import MixtureSameFamily
from .transformed_distribution import TransformedDistribution
from .tpp_utils import clamp_preserve_gradients


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time,
        std_log_inter_time: Std of log-inter-event-times,
    """

    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [
                D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)
            ]
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        transforms.append(D.ExpTransform())
        super().__init__(GMM, transforms)

    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()



class LogNormMix(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """

    def __init__(
        self,
        context_size=100,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        num_mix_components=128,
    ):
        super().__init__()
        self.context_size = context_size
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.num_mix_components = num_mix_components
        self.linear = nn.Linear(self.context_size, 3 * self.num_mix_components)

    def get_inter_time_dist(
        self, context: torch.Tensor
    ) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.linear(
            context
        )  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., : self.num_mix_components]
        log_scales = raw_params[
            ..., self.num_mix_components : (2 * self.num_mix_components)
        ]
        log_weights = raw_params[..., (2 * self.num_mix_components) :]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        return LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time,
        )

# =============================================================================
class Time2Vec(nn.Module):
    def __init__(self, activation, time_emb_size):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.f = torch.sin
        elif activation == "cos":
            self.f = torch.cos
        self.time_emb_size = time_emb_size
        self.tau_to_emb_aperiodic = nn.Linear(1, 1)
        self.tau_to_emb_periodic = nn.Linear(1, self.time_emb_size - 1)
        self.W = nn.parameter.Parameter(torch.randn(self.time_emb_size))
        # self.fc1 = nn.Linear(hiddem_dim, 2)

    def forward(self, tau):
        # x = x.unsqueeze(1)
        ## tau shape will be batch_size* seq_len
        batch_size, seq_len = tau.size()
        # tau = tau.view(-1,1)
        tau = tau.unsqueeze(-1)
        tau_ap = self.tau_to_emb_aperiodic(tau)  ## batch_size*seq_len*time_emb
        tau_p = self.f(self.tau_to_emb_periodic(tau))
        # tau_p = torch.sin(self.tau_to_emb_periodic(tau)) + torch.cos(self.tau_to_emb_periodic(tau))
        tau = torch.cat([tau_ap, tau_p], axis=-1)
        tau = tau * self.W
        return tau


class CondEventLSTM(nn.Module):
    def __init__(
        self,
        vocab,
        nb_layers,
        mean_log_inter_time,
        std_log_inter_time,
        nb_lstm_units=100,
        embedding_dim=3,
        time_emb_dim=10,
        device="cpu",
        num_mix_components=128,
        num_labels=10
    ):
        super(CondEventLSTM, self).__init__()
        self.vocab = vocab
        self.nb_lstm_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.time_emb_dim = time_emb_dim
        self.time_context_dim = embedding_dim
        # don't count the padding tag for the classifier output
        self.nb_events = len(self.vocab) - 1

        # when the model is bidirectional we double the output dimension
        # self.lstm
        # build embedding layer first
        nb_vocab_words = len(self.vocab)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.vocab["<PAD>"]
        self.word_embedding = nn.Embedding(
            num_embeddings=nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx,
        )
        self.t2v = Time2Vec("sin", self.time_emb_dim)
        #
        self.conditional_embedding = nn.Embedding(
            num_embeddings=num_labels+1,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx,
        )
        self.fc_h = nn.Linear(self.embedding_dim, self.nb_lstm_units)
        self.fc_c = nn.Linear(self.embedding_dim, self.nb_lstm_units)
        
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim + self.time_emb_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )  # dropout

        # output layer which projects back to tag space
        self.hidden_to_events = nn.Linear(self.nb_lstm_units, self.nb_events)
        # self.hiddent_
        self.hidden_to_hidden_time = nn.Linear(
            self.nb_lstm_units + self.embedding_dim, self.time_context_dim
        )
        
        self.sigmactivation = nn.Sigmoid()
        self.device = device
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.lognormalmix = LogNormMix(
            self.time_context_dim, self.mean_log_inter_time, self.std_log_inter_time, num_mix_components
        )

    def init_hidden(self, label_embs):
        h0 = self.fc_h(label_embs).unsqueeze(0).repeat(self.nb_lstm_layers, 1, 1)
        c0 = self.fc_c(label_embs).unsqueeze(0).repeat(self.nb_lstm_layers, 1, 1)
        return (h0, c0)

    def forward(self, X, Y, Xt, Yt, XDelta, YDelta, seq_labels, X_lengths, mask):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        label_embs = self.conditional_embedding(seq_labels)
        self.hidden = self.init_hidden(label_embs)
        
        batch_size, seq_len = X.size()
        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embedding(X)
        Y = self.word_embedding(Y)
        
        Xt = self.t2v(Xt)
        X = torch.cat((X, Xt), -1)
        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        X_lengths = X_lengths.cpu()
        X = torch.nn.utils.rnn.pack_padded_sequence(
            X, X_lengths, batch_first=True, enforce_sorted=False
        )

        # now run through LSTM
        # print(X.shape)
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, total_length=seq_len)
        # print(X.shape)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        Y = Y.view(-1, Y.shape[2])
        # print(X.shape,Y.shape)

        # run through actual Event linear layer
        Y_hat = self.hidden_to_events(X)
        Y_hat = Y_hat.view(batch_size, seq_len, self.nb_events)

        X = torch.cat((X, Y), -1)
        X = self.sigmactivation(X)
        X = self.hidden_to_hidden_time(X)
        X = self.sigmactivation(X)
        X = X.view(batch_size, seq_len, X.shape[-1])
        ##print("Here ,",X.shape)
        self.inter_time_dist = self.lognormalmix.get_inter_time_dist(
            X
        )  #### X is context
        YDelta = YDelta.clamp(1e-10)
        inter_time_log_loss = self.inter_time_dist.log_prob(
            YDelta
        )  ### batch_size*seq_len
        return (
            Y_hat,
            inter_time_log_loss,
        )  # ,T_hat


def get_event_prediction_rate(Y, Y_hat):
    ### Assumes pad in Y is -1
    ### Y_hat is unnormalized weights
    mask = Y != -1
    num_events = mask.sum()
    Y_hat = torch.argmax(Y_hat, dim=1)
    true_predicted = (Y_hat == Y) * mask
    true_predicted = true_predicted.sum()
    return true_predicted.item() * 1.00 / num_events.item()


def get_time_mse(T, T_hat, Y):
    mask = Y != -1
    num_events = mask.sum()
    diff = (T - T_hat) * mask
    diff = diff * diff
    return diff.sum() / num_events


def get_topk_event_prediction_rate(Y, Y_hat, k=5, ignore_Y_value=-1):
    ### Assumes pad in Y is -1
    ### Y_hat is unnormalized weights
    mask = Y != ignore_Y_value
    num_events = mask.sum().item()
    Y_topk = torch.topk(Y_hat, k=k, dim=-1, largest=True)
    Y_topk = Y_topk.indices.detach().cpu().numpy()
    Y_cpu = Y.detach().cpu().numpy()
    true_predicted = sum(
        [1 for i, item in enumerate(Y_cpu) if item != -1 and item in Y_topk[i]]
    )
    return true_predicted * 1.00 / num_events
