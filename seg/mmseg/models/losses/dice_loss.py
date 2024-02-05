# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add debug_output

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES    

@LOSSES.register_module()
class DiceLoss(nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self, 
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super().__init__()

        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = None #get_class_weight(class_weight)

        self.debug = False
        self.debug_output = None

        self.smooth = 1e-5
        self.eps = 0

    # def forward(self, logits, targets, reduction='mean', smooth=1e-6):
    def forward(self,
                logits,
                targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """

        num_classes = logits.shape[1]

        probabilities = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # prredicted probability for the actual class.

        intersection = (targets_one_hot * probabilities).sum(2).sum(2).sum(0)
        
        mod_a = probabilities.sum(2).sum(2).sum(0)
        mod_b = targets_one_hot.sum(2).sum(2).sum(0)       

        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (mod_a + mod_b  + self.eps + self.smooth)).mean()

        if self.debug:
            self.debug_output = {
                'Seg. Pred.': logits.detach().cpu().numpy(),
                'Seg. GT': targets.detach().cpu().numpy()
            }

        return self.loss_weight * dice_loss
    