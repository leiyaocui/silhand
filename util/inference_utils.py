import torch
import torch.nn.functional as F


def crop_masks(
    masks: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    source_image_sizes: torch.Tensor,
    valid_views: torch.Tensor,
    image_size: int = 224,
    bbox_area_fraction: float = 0.25,
) -> dict[str, torch.Tensor]:
    target_height = target_width = image_size
    batch_size, num_views, height, width = masks.shape
    device = masks.device
    flat_image_sizes = source_image_sizes.flatten(0, 1).float()
    source_height, source_width = flat_image_sizes.unbind(-1)
    pixels_y = torch.arange(height, device=device)
    pixels_x = torch.arange(width, device=device)
    foreground_masks = masks.flatten(0, 1) > 0.5
    foreground_masks &= (pixels_y[None, :, None] < source_height[:, None, None]) & (
        pixels_x[None, None, :] < source_width[:, None, None]
    )
    foreground_columns, foreground_rows = (
        foreground_masks.any(dim=1),
        foreground_masks.any(dim=2),
    )
    valid_masks = foreground_columns.any(dim=1) & valid_views.flatten()
    bbox_left = torch.where(foreground_columns, pixels_x, width).amin(dim=1).float()
    bbox_top = torch.where(foreground_rows, pixels_y, height).amin(dim=1).float()
    bbox_right = torch.where(foreground_columns, pixels_x + 1, 0).amax(dim=1).float()
    bbox_bottom = torch.where(foreground_rows, pixels_y + 1, 0).amax(dim=1).float()
    # Empty views use the full frame, without borrowing another view or GT box.
    bbox_left = torch.where(valid_masks, bbox_left, 0.0)
    bbox_top = torch.where(valid_masks, bbox_top, 0.0)
    bbox_right = torch.where(valid_masks, bbox_right, source_width)
    bbox_bottom = torch.where(valid_masks, bbox_bottom, source_height)
    bbox_width, bbox_height = bbox_right - bbox_left, bbox_bottom - bbox_top
    bbox_area_fraction = torch.full_like(bbox_left, bbox_area_fraction)
    target_aspect_ratio = target_width / target_height
    crop_height = torch.maximum(
        torch.sqrt(
            bbox_width * bbox_height / (bbox_area_fraction * target_aspect_ratio)
        ),
        torch.maximum(bbox_height, bbox_width / target_aspect_ratio),
    )
    crop_width = torch.maximum(crop_height * target_aspect_ratio, bbox_width)
    # Match the target aspect ratio when the sensor allows it, otherwise pad.
    crop_width = torch.minimum(crop_width, source_width)
    crop_height = torch.minimum(crop_height, source_height)
    crop_left = (bbox_left + bbox_right - crop_width) / 2.0
    crop_top = (bbox_top + bbox_bottom - crop_height) / 2.0
    crop_left = crop_left.clamp(
        min=torch.maximum(bbox_right - crop_width, torch.zeros_like(bbox_left)),
        max=torch.minimum(bbox_left, source_width - crop_width),
    )
    crop_top = crop_top.clamp(
        min=torch.maximum(bbox_bottom - crop_height, torch.zeros_like(bbox_top)),
        max=torch.minimum(bbox_top, source_height - crop_height),
    )
    resize_scale = torch.minimum(target_width / crop_width, target_height / crop_height)
    resized_width = (resize_scale * crop_width).round().to(torch.int32)
    resized_height = (resize_scale * crop_height).round().to(torch.int32)
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    crop_affines = torch.eye(3, device=device).expand(len(bbox_left), 3, 3).clone()
    crop_affines[:, 0, 0] = resize_scale
    crop_affines[:, 1, 1] = resize_scale
    crop_affines[:, 0, 2] = (
        pad_x - resize_scale * crop_left + (resize_scale - 1.0) / 2.0
    )
    crop_affines[:, 1, 2] = pad_y - resize_scale * crop_top + (resize_scale - 1.0) / 2.0
    output_pixels_x = torch.arange(target_width, device=device).float() + 0.5
    output_pixels_y = torch.arange(target_height, device=device).float() + 0.5
    source_pixels_x = (output_pixels_x[None] - pad_x[:, None]) / resize_scale[
        :, None
    ] + crop_left[:, None]
    source_pixels_y = (output_pixels_y[None] - pad_y[:, None]) / resize_scale[
        :, None
    ] + crop_top[:, None]
    grid_x = (2.0 * source_pixels_x / width - 1.0)[:, None].expand(
        -1, target_height, -1
    )
    grid_y = (2.0 * source_pixels_y / height - 1.0)[:, :, None].expand(
        -1, -1, target_width
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)
    valid_pixels = (
        (output_pixels_x[None, None] >= pad_x[:, None, None])
        & (output_pixels_x[None, None] < (pad_x + resized_width)[:, None, None])
        & (output_pixels_y[None, :, None] >= pad_y[:, None, None])
        & (output_pixels_y[None, :, None] < (pad_y + resized_height)[:, None, None])
        & valid_masks[:, None, None]
    )
    cropped_masks = F.grid_sample(
        foreground_masks[:, None].float(), grid, mode="nearest", align_corners=False
    )[:, 0]
    cropped_masks *= valid_pixels
    valid_masks &= cropped_masks.flatten(1).any(dim=1)
    with torch.autocast(device_type=device.type, enabled=False):
        cropped_cam_intrinsics = crop_affines @ cam_intrinsics.flatten(0, 1).float()
    return {
        "masks": cropped_masks.reshape(
            batch_size, num_views, target_height, target_width
        ),
        "cam_intrinsics": cropped_cam_intrinsics.reshape(batch_size, num_views, 3, 3),
        "valid_views": valid_masks.reshape(batch_size, num_views),
        "crop_boxes": torch.stack(
            (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height),
            -1,
        ).reshape(batch_size, num_views, 4),
        "crop_affines": crop_affines.reshape(batch_size, num_views, 3, 3),
    }
