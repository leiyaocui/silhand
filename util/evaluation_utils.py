import torch


def _similarity_align_points(pred_points, gt_points):
    output_dtype = pred_points.dtype
    pred_points, gt_points = pred_points.double(), gt_points.double()
    pred_mean, gt_mean = (
        pred_points.mean(-2, keepdim=True),
        gt_points.mean(-2, keepdim=True),
    )
    pred_centered, gt_centered = pred_points - pred_mean, gt_points - gt_mean
    u, singular_values, vh = torch.linalg.svd(
        pred_centered.transpose(-1, -2) @ gt_centered
    )
    correction = torch.ones_like(singular_values)
    correction[..., -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0)
    rotation = (u * correction.unsqueeze(-2)) @ vh
    score = (singular_values * correction).sum(-1)
    variance = pred_centered.square().sum((-2, -1))
    scale = score / variance.clamp_min(torch.finfo(pred_points.dtype).eps)
    return (scale[..., None, None] * (pred_centered @ rotation) + gt_mean).to(
        output_dtype
    )


@torch.inference_mode()
def benchmark_metrics(inputs, predictions):
    pred_vertices, pred_joints = (
        predictions["vertices_world"],
        predictions["joints_world"],
    )
    gt_vertices = inputs["gt_vertices"].to(pred_vertices)
    gt_joints = inputs["gt_joint_positions"].to(pred_joints)
    metrics = {}
    for metric_name, predicted, target in (
        ("mpjpe_mm", pred_joints, gt_joints),
        ("mpvpe_mm", pred_vertices, gt_vertices),
    ):
        pairs = (
            ("", predicted, target),
            ("rr_", predicted - pred_joints[:, :1], target - gt_joints[:, :1]),
            ("pa_", _similarity_align_points(predicted, target), target),
        )
        for prefix, first, second in pairs:
            metrics[prefix + metric_name] = (first - second).norm(p=2, dim=-1).mean(
                -1
            ) * 1000.0
    return metrics
