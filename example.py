import numpy as np
from mean_average_precision import MetricBuilder

# print list of available metrics
print(MetricBuilder.get_metrics_list())

num_classes = 5
# create metric_fn
metric_fn = MetricBuilder.build_evaluation_metric(
    "map_2d", async_mode=True, num_classes=num_classes
)

# add some samples to evaluation
for i in range(num_classes):
    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    gt = np.array(
        [
            [439, 157, 556, 241, i, 0, 0],
            [437, 246, 518, 351, i, 0, 0],
            [515, 306, 595, 375, i, 0, 0],
            [407, 386, 531, 476, i, 0, 0],
            [544, 419, 621, 476, i, 0, 0],
            [609, 297, 636, 392, i, 0, 0],
        ]
    )

    # [xmin, ymin, xmax, ymax, class_id, confidence]
    preds = np.array(
        [
            [429, 219, 528, 247, i, 0.460851],
            [433, 260, 506, 336, i, 0.269833],
            [518, 314, 603, 369, i, 0.462608],
            [592, 310, 634, 388, i, 0.298196],
            [403, 384, 517, 461, i, 0.382881],
            [405, 429, 519, 470, i, 0.369369],
            [433, 272, 499, 341, i, 0.272826],
            [413, 390, 515, 459, i, 0.619459],
        ]
    )

    metric_fn.add(preds, gt)

# compute PASCAL VOC metric
print(
    f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))}"
)

# compute PASCAL VOC metric at the all points
print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)}")
