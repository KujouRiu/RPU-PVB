import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import RandomNegativeTripletSelector


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector=None, **kwargs):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector if triplet_selector else RandomNegativeTripletSelector(margin,
                                                                                                        **kwargs)

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if len(triplets) == 0:
            return torch.tensor(0.1).to(embeddings.device), 0

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        m = 0.65
        gamma = 256
        # ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        # ap_distances.size -----> torch.Size([240])
        # an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)

        # print(embeddings[triplets[:, 0]])
        # torch.Size([240, 4])
        # ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]])  # .pow(.5)
        # an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]])  # .pow(.5)
        ap_distances_cos = torch.cosine_similarity(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]], dim=0)
        an_distances_cos = torch.cosine_similarity(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]], dim=0)
        Softmax = torch.nn.Softmax(dim=0)

        ap = torch.clamp_min(- ap_distances_cos.detach() + 1 + m, min=0.)
        an = torch.clamp_min(an_distances_cos.detach() + m, min=0.)
        delta_p = 1 - m
        delta_n = m
        logit_p = - ap * (ap_distances_cos - delta_p) * gamma
        logit_n = an * (an_distances_cos - delta_n) * gamma

        losses = Softmax(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        # losses = F.relu(ap_distances - an_distances + self.margin)

        # losses = F.relu(ap_distances - an_distances + self.margin)
        return losses, len(triplets)
        # return losses.mean(), len(triplets)
