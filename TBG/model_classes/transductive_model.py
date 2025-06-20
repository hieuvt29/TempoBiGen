import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

from torch.distributions import Categorical
import torch.distributions as D

from .normal import Normal
from .mixture_same_family import MixtureSameFamily
from .transformed_distribution import TransformedDistribution
from .tpp_utils import clamp_preserve_gradients

EPS = 1e-6 

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


class NormalMixtureDistribution(TransformedDistribution):
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
        mean_: float = 0.0,
        std_: float = 1.0,
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_ == 0.0 and std_ == 1.0:
            transforms = []
        else:
            transforms = [
                D.AffineTransform(loc=mean_, scale=std_)
            ]
        self.mean_ = mean_
        self.std_ = std_
        super().__init__(GMM, transforms)

    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.
        """
        a = self.std_
        b = self.mean_
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

    # modify the string format of the class to include context_size, num_mix_components, linear layer
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"\n    context_size={self.context_size},"
            f"\n    mean_log_inter_time={self.mean_log_inter_time:.2f},"
            f"\n    std_log_inter_time={self.std_log_inter_time:.2f},"
            f"\n    num_mix_components={self.num_mix_components},"
            f"\n    linear={self.linear}"
            "\n)"
        )
        
    
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


class HiddenToEvents(nn.Module):
    def __init__(self, nb_lstm_units, nb_events, fp_size, sp_size, vocab):
        super(HiddenToEvents, self).__init__()
        self.vocab = vocab
        self.nb_lstm_units = nb_lstm_units
        self.fp_size = fp_size
        self.sp_size = sp_size
        
        self.hidden_to_end = nn.Linear(self.nb_lstm_units, 1)
        self.hidden_to_hcw = nn.Linear(self.nb_lstm_units, self.fp_size)
        self.hidden_to_roo = nn.Linear(self.nb_lstm_units, self.sp_size) # +1 for the padding tag we don't count
        
    def log_prob(self, X, pY, Y):
        # X: (batch_size, seq_len, nb_lstm_units)
        # pX: (batch_size, seq_len,)
        
        batch_size, seq_len = X.shape[:2]
        pY = pY.reshape(batch_size, seq_len)
        X = X.view(batch_size * seq_len, -1)
        Y = Y.view(batch_size * seq_len)
        pY = pY.view(-1)
        # create two input tensors for the two different linear layers   
        end_pad_pos = pY == 0 # both end node and padding
        hcw_pos = pY == 1
        roo_pos = pY == 2
        
        # X_hcw = X[(pX == 1) & ~end_pad_pos] # pY == 2
        # X_roo = X[(pX == 2) & ~end_pad_pos] # pY == 1
        
        # logits_hcw = self.hidden_to_hcw(X_roo)
        # logits_roo = self.hidden_to_roo(X_hcw)
        prob = torch.ones(batch_size * seq_len).to(X.device)
        prob_all_mat = torch.zeros(batch_size * seq_len, self.fp_size + self.sp_size + 2).to(X.device)
        
        end_node_logit = self.hidden_to_end(X)
        end_node_prob = torch.sigmoid(end_node_logit).squeeze(-1).to(prob.dtype)
        non_end_prob = 1 - end_node_prob
        
        prob[end_pad_pos] = end_node_prob[end_pad_pos]
        prob_all_mat[end_pad_pos, 0:2] = end_node_prob[end_pad_pos].unsqueeze(-1) # both end node and padding are the same logit
        
        if hcw_pos.sum() > 0:
            logits_hcw = self.hidden_to_hcw(X[hcw_pos])
            P_hcw = torch.softmax(logits_hcw, dim=-1)
            Y_hcw = Y[hcw_pos]
            prob_hcw = P_hcw[range(len(P_hcw)), Y_hcw - 2]
            prob[hcw_pos] = prob_hcw * non_end_prob[hcw_pos]
            prob_all_mat[hcw_pos, 2:2+self.fp_size] = P_hcw * non_end_prob[hcw_pos].unsqueeze(-1)
            
        if roo_pos.sum() > 0:
            logits_roo = self.hidden_to_roo(X[roo_pos])
            P_roo = torch.softmax(logits_roo, dim=-1)
            Y_roo = Y[roo_pos]
            prob_roo = P_roo[range(len(P_roo)), Y_roo - 2 - self.fp_size]
            prob[roo_pos] = prob_roo * non_end_prob[roo_pos]
            prob_all_mat[roo_pos, 2+self.fp_size:] = P_roo * non_end_prob[roo_pos].unsqueeze(-1)
            
        # compute the log likelihood of the true events
        log_prob = torch.log(prob)
        
        return log_prob.view(batch_size, seq_len), prob_all_mat.view(batch_size, seq_len, -1)
    
    def sample(self, X, pX):
        # X: (batch_size, seq_len, nb_lstm_units)
        # pX: (batch_size, seq_len,)
        batch_size, seq_len = X.shape[:2]
        pX = pX.reshape(batch_size, seq_len)
        pX = pX.view(-1)
        X = X.view(batch_size * seq_len, -1)
        
        #
        pY = torch.zeros(batch_size, seq_len).to(X.device)
        pY = pY.view(batch_size * seq_len)
        pY[pX == 1] = torch.tensor(2)
        pY[pX == 2] = torch.tensor(1)
        
        end_node_logits = self.hidden_to_end(X)
        end_node_probs = torch.sigmoid(end_node_logits).squeeze(-1)
        end_node = D.Bernoulli(end_node_probs).sample().to(pY.dtype) # may over
        pY[end_node == 1] = torch.tensor(0)
        
        end_pad_pos = pY == 0 # both end node and padding
        hcw_pos = pY == 1
        roo_pos = pY == 2
        
        sampled_Y = torch.zeros(batch_size * seq_len, dtype=torch.long).to(X.device)
        sampled_Y = sampled_Y.view(-1)
        
        if end_pad_pos.sum() > 0:
            sampled_Y[end_pad_pos] = torch.tensor(self.vocab['end_node'], dtype=sampled_Y.dtype)
            
        if hcw_pos.sum() > 0:
            logits_hcw = self.hidden_to_hcw(X[hcw_pos])
            P_hcw = torch.softmax(logits_hcw, dim=-1)
            sampled_Y_hcw = D.Categorical(probs=P_hcw).sample().to(sampled_Y.dtype) + 2
            sampled_Y[hcw_pos] = sampled_Y_hcw
        
        if roo_pos.sum() > 0:
            logits_roo = self.hidden_to_roo(X[roo_pos])
            P_roo = torch.softmax(logits_roo, dim=-1)
            sampled_Y_roo = D.Categorical(probs=P_roo).sample().to(sampled_Y.dtype) + 2 + self.fp_size
            sampled_Y[roo_pos] = sampled_Y_roo
        
        return sampled_Y.view(batch_size, seq_len), pY


class CondBipartiteEventLSTM(nn.Module):
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
        num_labels=10, 
        fp_size=5,
        sp_size=5
    ):
        super(CondBipartiteEventLSTM, self).__init__()
        self.vocab = vocab
        self.nb_lstm_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.time_emb_dim = time_emb_dim
        self.time_context_dim = embedding_dim
        # don't count the padding tag for the classifier output
        self.nb_events = len(self.vocab) - 1
        self.nb_labels = num_labels
        self.register_buffer("fp_size", torch.tensor(fp_size))
        self.register_buffer("sp_size", torch.tensor(sp_size))
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
        self.label_embedding = nn.Embedding(
            num_embeddings=num_labels+1,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx,
        )
        self.fc_init_h0 = nn.Linear(self.embedding_dim, self.nb_lstm_units)
        self.fc_init_h1 = nn.Linear(self.embedding_dim, self.nb_lstm_units)
        
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim + self.time_emb_dim, # prev node emb + time emb
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )  # dropout

        # output layer which projects back to tag space
        self.hidden_to_events = HiddenToEvents(self.nb_lstm_units, self.nb_events, fp_size, sp_size, self.vocab)
        # self.hiddent_
        self.hidden_to_hidden_time = nn.Sequential(
            nn.Linear(
                self.nb_lstm_units + self.embedding_dim, self.time_context_dim
            ), 
            nn.Sigmoid()
        )
        
        # self.sigmactivation = nn.Sigmoid()
        self.device = device
        
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.lognormalmix = LogNormMix(
            self.time_context_dim, self.mean_log_inter_time, 
            self.std_log_inter_time, num_mix_components
        )
        
    def init_hidden(self, label_embs):
        h0 = self.fc_init_h0(label_embs).unsqueeze(0).repeat(self.nb_lstm_layers, 1, 1)
        c0 = self.fc_init_h1(label_embs).unsqueeze(0).repeat(self.nb_lstm_layers, 1, 1)
        return (h0, c0)

    def forward(self, X, Y, Xt, Yt, XDelta, YDelta,
                seq_labels, pX, pY, X_lengths, mask):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        label_embs = self.label_embedding(seq_labels)
        self.hidden = self.init_hidden(label_embs)
        Y_label = Y.clone()
        batch_size, seq_len = X.size()
        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.word_embedding(X)
        Y = self.word_embedding(Y)
        
        Xt = self.t2v(Xt)
        Yt = self.t2v(Yt)

        ####
        #### Node Infernece
        ####
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
        
        # ---------------------
        event_log_prob, Y_hat = self.hidden_to_events.log_prob(X, pY, Y_label)
        
        ####
        #### Time Inference
        ####
        X = torch.cat((X, Y), -1)
        X_dt = F.sigmoid(X)
        X_dt = self.hidden_to_hidden_time(X_dt)
        X_dt = X_dt.view(batch_size, seq_len, X_dt.shape[-1])
        self.inter_time_dist = self.lognormalmix.get_inter_time_dist(
            X_dt
        )  #### X is context
        YDelta = YDelta.clamp(1e-10) # avoid 0 values
        inter_time_log_loss = self.inter_time_dist.log_prob(
            YDelta
        )  ### batch_size, seq_len
                
        return (
            event_log_prob,
            inter_time_log_loss, Y_hat
        )  # ,T_hat
    
    
    def get_pX(self, X):
        pX = torch.zeros_like(X)
        pX[X <= 1] = 0 # end_node and padding
        pX[(X >= 2) & (X < 2 + self.fp_size)] = 1 # hcw
        pX[(X >= 2 + self.fp_size)] = 2 # room
        return pX
        
    def sample(self, init_X, init_Xt, init_pX, init_seq_labels, max_seq_len=10):
        # init_X: (batch_size, seq_len)
        # init_Xt: (batch_size, seq_len)
        # init_seq_labels: (batch_size,)
        
        label_embs = self.label_embedding(init_seq_labels)
        self.hidden = self.init_hidden(label_embs)
        batch_size, seq_len = init_X.size()
        length = 0
        batch_generated_events = [init_X.detach().cpu().numpy()]
        batch_generated_times = [init_Xt.detach().cpu().numpy()]
        batch_generated_labels = [init_seq_labels.detach().cpu().numpy()]
        
        
        while length < max_seq_len:
            length += 1
            X = self.word_embedding(init_X)
            Xt = self.t2v(init_Xt)
            X = torch.cat((X, Xt), -1)
            X, self.hidden = self.lstm(X, self.hidden)
            #
            Y_hat, pY = self.hidden_to_events.sample(X, init_pX)
            Y_hat = Y_hat.view(-1, Y_hat.shape[-1])
            # Y_hat = Y_hat + 1 # True labels are shifted by 1 (old code)
            Y = self.word_embedding(Y_hat)
            #
            X = torch.cat((X, Y), -1)
            X_dt = F.sigmoid(X)
            X_dt = self.hidden_to_hidden_time(X_dt)
            X_dt = X_dt.view(batch_size, seq_len, X_dt.shape[-1])
            
            inter_time_dist = self.lognormalmix.get_inter_time_dist(X_dt)
            YDelta_hat = inter_time_dist.sample().view(batch_size, seq_len)
            Yt_hat = YDelta_hat.add(init_Xt).round()
            Yt_hat[Yt_hat < 1] = 1
            #
            init_X = Y_hat
            init_Xt = Yt_hat
            init_pX = pY
            #
            batch_generated_events.append(init_X.detach().cpu().numpy())
            batch_generated_times.append(init_Xt.detach().cpu().numpy())
            
        return (
            batch_generated_events,
            batch_generated_times
        )
        
        
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
