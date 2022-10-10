import torch
import torch.nn.functional as F


def cosine_similarity_2d(a, b):
    """

    Parameters
    ----------
    a: torch.Tensor
        shape=[N1, D]
    b: torch.Tensor
        shape=[N2, D]

    Returns
    -------
    cos: torch.Tensor
        shape=[N1, N2]
    """
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[1]
    a_norm = F.normalize(a, dim=1, eps=1e-8)
    b_norm = F.normalize(b, dim=1, eps=1e-8)
    cos = torch.matmul(a_norm, b_norm.t())
    return cos


def l2_similarity_2d(a, b):
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = a.unsqueeze(1) # N1, 1, D
    b = b.unsqueeze(0) # 1, N2, D
    diff = a - b
    l2 = torch.norm(diff, dim=2)
    return l2


def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (b, N, 4) ndarray of float
    gt_boxes: (b, N, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)
    N = anchors.size(1)
    K = gt_boxes.size(1)
    assert N == K

    gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
    gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
    gt_boxes_area = (gt_boxes_x * gt_boxes_y)

    anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
    anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
    anchors_area = (anchors_boxes_x * anchors_boxes_y)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

    # boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
    # query_boxes = gt_boxes.view(batch_size, 1, K, 5).expand(batch_size, N, K, 5)

    iw = (torch.min(anchors[:, :, 2], gt_boxes[:, :, 2]) -
          torch.max(anchors[:, :, 0], gt_boxes[:, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(anchors[:, :, 3], gt_boxes[:, :, 3]) -
          torch.max(anchors[:, :, 1], gt_boxes[:, :, 1]) + 1)
    ih[ih < 0] = 0
    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    overlaps = overlaps * (1 - anchors_area_zero.float())
    overlaps[overlaps < 0] = 0


    return overlaps