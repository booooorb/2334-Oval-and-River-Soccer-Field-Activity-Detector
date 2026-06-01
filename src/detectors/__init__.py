"""Detector model registry."""

from src.detectors import (
    balanced_previous_diff_blur,
    balanced_previous_diff_blur_lumps,
    balanced_previous_diff_blur_lumps_deploy,
    blur_stabilized_balanced_previous_diff,
    gmm_mog2_foreground_motion,
    hsv_balanced_previous_diff_blur,
    hsv_blue_diff,
    hsv_local_support,
    hsv_pbas,
    hsv_previous_diff,
    hsv_running_gaussian,
    lab_bilateral_previous_diff,
    tiny_cnn_activity,
)

MODEL_EVALUATORS = {
    hsv_previous_diff.MODEL_NAME: hsv_previous_diff.evaluate,
    hsv_pbas.MODEL_NAME: hsv_pbas.evaluate,
    hsv_local_support.MODEL_NAME: hsv_local_support.evaluate,
    hsv_running_gaussian.MODEL_NAME: hsv_running_gaussian.evaluate,
    hsv_balanced_previous_diff_blur.MODEL_NAME: hsv_balanced_previous_diff_blur.evaluate,
    hsv_blue_diff.MODEL_NAME: hsv_blue_diff.evaluate,
    balanced_previous_diff_blur.MODEL_NAME: balanced_previous_diff_blur.evaluate,
    balanced_previous_diff_blur_lumps.MODEL_NAME: balanced_previous_diff_blur_lumps.evaluate,
    balanced_previous_diff_blur_lumps_deploy.MODEL_NAME: balanced_previous_diff_blur_lumps_deploy.evaluate,
    blur_stabilized_balanced_previous_diff.MODEL_NAME: blur_stabilized_balanced_previous_diff.evaluate,
    gmm_mog2_foreground_motion.MODEL_NAME: gmm_mog2_foreground_motion.evaluate,
    lab_bilateral_previous_diff.MODEL_NAME: lab_bilateral_previous_diff.evaluate,
    lab_bilateral_previous_diff.MODEL_NAME_COLOR: lab_bilateral_previous_diff.evaluate_color,
    lab_bilateral_previous_diff.MODEL_NAME_COLOR_STRONG: lab_bilateral_previous_diff.evaluate_color_strong,
    tiny_cnn_activity.MODEL_NAME: tiny_cnn_activity.evaluate,
}

MODEL_NAMES = tuple(MODEL_EVALUATORS)
