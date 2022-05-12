import matplotlib.pyplot as plt
import numpy as np

predicates_logit_indices = {'all': range(360), 'front_of': range(90), 'right_of': range(90, 180),
                            'contains': range(180, 270), 'supports': range(270, 360)}

def calculate_metrics(predictions, targets, masks):
    metrics = {}
    metrics['target_true'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        metrics['target_true'][predicate] = np.sum(targets[:, logit_indices] * masks[:, logit_indices]) / np.sum(masks[:, logit_indices]) * 100
    metrics['prediction_positive'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        metrics['prediction_positive'][predicate] = np.sum(predictions[:, logit_indices] * masks[:, logit_indices]) / np.sum(masks[:, logit_indices]) * 100
    metrics['predicate_accuracy'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        metrics['predicate_accuracy'][predicate] = np.sum((predictions[:, logit_indices] == targets[:, logit_indices]) * masks[:, logit_indices]) / np.sum(masks[:, logit_indices]) * 100
    metrics['scene_accuracy'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        a = np.sum((predictions[:, logit_indices] == targets[:, logit_indices]) * masks[:, logit_indices], axis=-1) / np.sum(masks[:, logit_indices], axis=-1)
        metrics['scene_accuracy'][predicate] = np.nansum(a) / np.sum(np.isreal(a)) * 100
    metrics['scene_all_accuracy'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        a = np.all((predictions[:, logit_indices] == targets[:, logit_indices]) | ~masks[:, logit_indices], axis=-1)
        metrics['scene_all_accuracy'][predicate] = np.nansum(a) / np.sum(np.isreal(a)) * 100
    metrics['predicate_precision'] = {}
    metrics['predicate_recall'] = {}
    metrics['predicate_f1'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        tp = ((predictions[:, logit_indices] & targets[:, logit_indices]) * masks[:, logit_indices]).sum()
        fp = ((predictions[:, logit_indices] & ~targets[:, logit_indices]) * masks[:, logit_indices]).sum()
        fn = ((~predictions[:, logit_indices] & targets[:, logit_indices]) * masks[:, logit_indices]).sum()
        precision = tp / (tp + fp) * 100
        recall = tp / (tp + fn) * 100
        f1 = 2 * precision * recall / (precision + recall)
        metrics['predicate_precision'][predicate] = precision
        metrics['predicate_recall'][predicate] = recall
        metrics['predicate_f1'][predicate] = f1
    return metrics


if __name__ == '__main__':
    # predictions = np.load('predictions.npy')
    # targets = np.load('targets.npy')
    # masks = np.load('masks.npy')
    tests_metrics = {}
    for dir, test_name in [('barplot/split_split_train/', 'cube_cube_train'),
                           ('barplot/split_split_valA/', 'cube_cube_valA'),
                           ('barplot/split_split_valB/', 'cube_cube_valB')]:
        tests_metrics[test_name] = calculate_metrics(np.load(dir + 'predictions.npy'),
                                                     np.load(dir + 'targets.npy'),
                                                     np.load(dir + 'masks.npy'))

    majority_predictions = np.zeros(np.load(dir + 'predictions.npy').shape, dtype=np.int8)
    for predicate, logit_indices in predicates_logit_indices.items():
        majority_predictions[logit_indices] = 1 if tests_metrics['cube_cube_valB']['target_true'][predicate] >= 50 else 0
    tests_metrics['majority_cube_valB'] = calculate_metrics(majority_predictions,
                                                            np.load(dir + 'targets.npy'),
                                                            np.load(dir + 'masks.npy'))

    width = 10
    dist = width * (len(tests_metrics) + 1)
    fig, axs = plt.subplots(3, 1)
    for i, (test, metrics) in enumerate(tests_metrics.items()):
        metrics_to_graph = {'scene_accuracy': metrics['scene_accuracy'],
                            'scene_all_accuracy': metrics['scene_all_accuracy'],
                            'predicate_f1': metrics['predicate_f1']}
        for j, (metric_name, metric) in enumerate(metrics_to_graph.items()):
            xs = np.arange(len(metric.keys())) * dist + width * i
            heights = [value for predicate, value in metric.items()]
            predicates = [predicate for predicate, value in metric.items()]
            bars = axs[j].bar(xs, heights, width, label=test)
            for k in range(len(xs)):
                axs[j].annotate(
                    f"{heights[k]:.1f}",
                    xy=(xs[k], heights[k]),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom'
                )
            axs[j].set_ylim([0, 150])
            axs[j].set_xticks(xs, metric.keys())
            axs[j].set_title(metric_name)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle('Metrics for geospa_cube dataset')
    plt.legend()
    plt.show()