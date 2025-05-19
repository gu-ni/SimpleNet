"""Anomaly metrics."""
import cv2
import numpy as np
from sklearn import metrics
from tqdm import tqdm

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    precision, recall, _ = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auc_pr = metrics.auc(recall, precision)
    
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


import pandas as pd
from skimage import measure
def compute_pro(masks, amaps, num_th=200):
    """
    Compute the Per-Region Overlap (PRO) score for pixel-level anomaly localization.

    Args:
        masks (np.ndarray): Binary ground-truth masks. Shape (N, H, W)
        amaps (np.ndarray): Anomaly maps (float), higher values indicate more anomalous. Shape (N, H, W)
        num_th (int): Number of threshold steps.

    Returns:
        pro_auc (float): Area under the PRO vs FPR curve.
    """
    records = []

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for th in tqdm(np.arange(min_th, max_th, delta)):
        binary_amaps = (amaps > th).astype(np.uint8)  # fresh binary map each iteration

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            dilated_amap = cv2.dilate(binary_amap, kernel)
            labeled_mask = measure.label(mask)
            props = measure.regionprops(labeled_mask)
            for region in props:
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = dilated_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        records.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th})

    df = pd.DataFrame.from_records(records)

    # Normalize FPR to 0~0.3 range
    df = df[df["fpr"] < 0.3]
    if not df.empty:
        df["fpr"] = df["fpr"] / df["fpr"].max()
        pro_auc = metrics.auc(df["fpr"], df["pro"])
    else:
        pro_auc = 0.0  # fallback if no valid region

    y_true = masks.flatten().astype(np.uint8)
    y_score = amaps.flatten().astype(np.float32)
    y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-6)
    if np.unique(y_true).size == 1:
        pixel_ap = 0.0  # or np.nan
    else:
        pixel_ap = metrics.average_precision_score(y_true, y_score)
    
    return pro_auc, pixel_ap

"""def compute_pro(masks, amaps, num_th=200):

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc"""
    
    
    
    


def compute_imagewise_retrieval_metrics_custom(anomaly_prediction_weights, anomaly_ground_truth_labels):
    """
    Computes image-wise AUROC.

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    Returns:
        dict with "auroc"
    """
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc}


def compute_pixelwise_retrieval_metrics_custom(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise AUROC and pixel-wise AP.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW]
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW]
    Returns:
        dict with "auroc" and "ap"
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    ap = metrics.average_precision_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    return {
        "ap": ap,
    }