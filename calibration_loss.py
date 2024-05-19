import torch
import torch.nn as nn

def safe_acos(x, eps=1e-7):
    return torch.acos(x.clamp(-1 + eps, 1 - eps))

# calibration loss with consistency loss regularisation
class CalibrationLoss(nn.Module):
    def __init__(self, weight=0.1):
        super(CalibrationLoss, self).__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss()

    def forward(self, embeddings, gt):
        # first compute the mse loss as embeddings will convert to unit vectors
        mse_loss = self.mse_loss(embeddings, gt)

        # Normalize the embeddings to unit vectors
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        # Compute cosine similarity matrix
        cosine_similarity_matrix = torch.matmul(embeddings, embeddings.T)


        # Clamp the cosine similarity values to avoid numerical issues with acos
        #cosine_similarity_matrix = torch.clamp(cosine_similarity_matrix, -1.0, 1.0)

        # Compute the angular distance matrix
        # compute the safe else nan values are coming
        angular_distance_matrix = safe_acos(cosine_similarity_matrix)#torch.acos(cosine_similarity_matrix)

        # Create a mask to exclude the diagonal elements (self-distances)
        mask = torch.eye(angular_distance_matrix.size(0), device=embeddings.device).bool()

        # Exclude the diagonal elements by applying the mask
        angular_distances = angular_distance_matrix[~mask].view(angular_distance_matrix.size(0), -1)

        # Compute the mean angular distance
        consistency_loss = torch.mean(angular_distances)

        return self.weight * consistency_loss + mse_loss