import argparse
import os
import pickle
import numpy as np

results_voc = {key: [[0.0] * 3, [0.0] * 3, [0.0] * 3, [0.0], [0.0, 0.0], [0.0] * 3, [0.0]] for key in
               ['alike-l', 'alike-n', 'alike-t', 'alike-s', 'superpoint']}
keys_net = list(results_voc.keys())


def log_extractor(dir, scene, measure_significant, net):
    significance_data = []
    for model in sorted(os.listdir(dir)):
        if model not in net:
            continue

        scene_path = os.path.join(dir, model, "results")
        try:
            newest_file = get_newest_result_file(scene_path)
            if not newest_file:
                print(f"No files found in {scene_path}")
                continue

            with open(newest_file, 'rb') as f:
                data = pickle.load(f)

            (init_dist_errors, init_angle_errors,
             estimated_dist_errors, estimated_angle_errors,
             *_) = data  # Unpack only necessary fields

            # from m to cm
            init_dist_errors = [x * 100 for x in init_dist_errors]
            estimated_dist_errors = [[x * 100 for x in y] for y in estimated_dist_errors]

            update_results(model, init_dist_errors, init_angle_errors, estimated_dist_errors, estimated_angle_errors)
            print_results(scene, model, init_dist_errors, init_angle_errors, estimated_dist_errors,
                          estimated_angle_errors, measure_significant)

            # consider only the 3rd iteration
            dist_errors = np.array(estimated_dist_errors[2])
            angle_errors = np.array(estimated_angle_errors[2])

            # Check where both conditions hold
            both_below_thresh = np.sum((dist_errors <= measure_significant[0]) &
                                       (angle_errors <= measure_significant[1]))

            total = len(dist_errors)
            percentage = (both_below_thresh / total) if total > 0 else 0.0

            significance_data.append((scene, percentage))

        except Exception as e:
            print(f"Error processing {scene_path}: {e}")

    return significance_data


def get_newest_result_file(path):
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.startswith("results_")]
    if not files:
        return None
    return os.path.join(path, sorted(files, key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)[0])


def update_results(model, init_dist_errors, init_angle_errors, estimated_dist_errors, estimated_angle_errors):
    dist_medians = [np.median(estimated_dist_errors[i]) for i in range(3)]
    angle_medians = [np.median(estimated_angle_errors[i]) for i in range(3)]

    for i in range(3):
        results_voc[model][0][i] += dist_medians[i]
        results_voc[model][1][i] += angle_medians[i]

    results_voc[model][2][0] += np.median(init_dist_errors)
    results_voc[model][2][1] += np.median(init_angle_errors)


def print_results(scene, model, init_dist_errors, init_angle_errors, estimated_dist_errors, estimated_angle_errors,
                  measure_significant):
    scene_name = scene.split("_")[0]
    model_name = "SuperPoint" if model == "superpoint" else model

    init_dist_median = np.median(init_dist_errors)
    init_angle_median = np.median(init_angle_errors)

    dist_medians = [np.median(estimated_dist_errors[i]) for i in range(3)]
    angle_medians = [np.median(estimated_angle_errors[i]) for i in range(3)]

    print(
        f"{scene_name} & 1st & {model_name} & {init_dist_median:.2f}/{init_angle_median:.2f} & {dist_medians[0]:.2f}/{angle_medians[0]:.2f}\\")
    print(f"             & 2nd &              &      -     & {dist_medians[1]:.2f}/{angle_medians[1]:.2f}\\")
    print(f"             & 3rd &              &      -     & {dist_medians[2]:.2f}/{angle_medians[2]:.2f}\\")
    print("\\cline{2-7}")


def print_results_per_network(results_voc, count, net):
    for key in net:
        dist_error_1, angle_error_1 = results_voc[key][0][0] / count, results_voc[key][1][0] / count
        dist_error_2, angle_error_2 = results_voc[key][0][1] / count, results_voc[key][1][1] / count
        dist_error_3, angle_error_3 = results_voc[key][0][2] / count, results_voc[key][1][2] / count
        init_dist_error, init_angle_error = results_voc[key][4][0] / count, results_voc[key][4][1] / count

        print(
            f"Cambridge & 1st & {key} & {init_dist_error:.2f}/{init_angle_error:.2f} & {dist_error_1:.2f} / {angle_error_1:.2f}\\")
        print(f"             & 2nd &              &      -     & {dist_error_2:.2f} / {angle_error_2:.2f}\\")
        print(f"             & 3rd &              &      -     & {dist_error_3:.2f} / {angle_error_3:.2f}\\")
        print("\\cline{2-7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--logs_dir', required=True,
        help='Logs file path (required)'
    )
    parser.add_argument(
        '--dataset', required=True,
        help='Dataset name (required) choose between Cambridge, 7Scenes'
    )
    parser.add_argument(
        '--net_model', type=str,
        help='Net model, choose between alike-(l,n,s,t), superpoint'
    )

    args = parser.parse_args()

    datasets = ['Cambridge', '7Scenes']
    for dataset in datasets:
        if dataset.lower() == args.dataset.lower():
            dataset_type = dataset
            break

    base_path = args.logs_dir
    net = args.net_model

    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    count = 0

    # check if net is a list
    if not isinstance(net, list):
        net = [net]

    print("Measures are in cm and degrees")
    measure_significant = {'Cambridge': [25, 2.0], '7Scenes': [5, 5.0]}

    sig_data = []
    for subdir in subdirs:
        sig_data_scene = log_extractor(os.path.join(base_path, subdir), subdir, measure_significant[dataset_type], net)
        sig_data.extend(sig_data_scene)
        count += 1

    for scene, percentage in sig_data:
        print(
            f"{scene}\t 3rd iter, <{measure_significant[dataset_type][0]:.0f} cm && <{measure_significant[dataset_type][1]:.0f} deg: {percentage:.2f}\%\\\\")

    print(f"Average results per network:")
    print_results_per_network(results_voc, count, net)
