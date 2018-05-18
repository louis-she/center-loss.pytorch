import torch

def compute_center_loss(features, centers, targets, lamda):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    center_loss = lamda / 2 * torch.sum(torch.pow(features - target_centers, 2)).item()
    return center_loss

def get_center_delta(features, centers, targets, alpha):
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = features - target_centers
    uni_targets, indices = torch.unique( targets, sorted=True, return_inverse=True )
    delta_centers = torch.zeros(uni_targets.size(0), delta_centers.size(1)).index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result