import argparse
import os
import re

import numpy as np


plt = None
DEFAULT_EXACT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_exact_cache")

# Nature-style visual system shared with campaign/code/plot_fig3_frustrated_v5.py.
SURFACE, INK, INK2, MUTED = "#ffffff", "#000000", "#4d4d4d", "#8c8c8c"
RULE = "#b3b3b3"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_COLORS = [
    BLUE,
    ORANGE,
    GREEN,
    "#8f63b8",
    "#d49b24",
    "#3d9fbf",
    "#cc5a71",
    "#6f7f45",
    "#73564a",
]
FONTS = ["Arial", "Helvetica", "DejaVu Sans"]


def apply_figure_style():
    plt.rcParams.update(
        {
            "font.family": FONTS,
            "mathtext.fontset": "dejavusans",
            "font.size": 6,
            "axes.labelsize": 6.5,
            "axes.labelcolor": INK,
            "axes.titlesize": 6.5,
            "axes.titleweight": "normal",
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": SURFACE,
        }
    )


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(RULE)
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(
        colors=INK2,
        labelsize=9,
        width=0.5,
        length=2.2,
        pad=1.8,
    )
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)

def import_matplotlib():
    try:
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it first, for example: "
            "pip install matplotlib"
        ) from exc
    return pyplot


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare saved Diffushadow evaluation .npz files.")
    parser.add_argument(
        "--predict_model",
        "--model",
        dest="predict_model",
        type=str,
        required=True,
        choices=["TFI", "J1J2", "ANNNI"],
        help="Physical model of the input .npz files.",
    )
    parser.add_argument(
        "--files",
        metavar="FILE",
        type=str,
        nargs="+",
        required=True,
        help="List of .npz files to compare.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=False,
        help="Labels for each file being compared. Must match the number of files.",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="all",
        choices=[
            "all",
            "zz",
            "xs",
            "energy",
            "xx",
            "yy",
            "spin_dot",
            "dimer",
            "x",
            "sf",
            "sf_pi",
            "sf_pi_over_2",
        ],
        help=(
            "Plot all properties or one property. TFI: zz/xs/energy; "
            "J1J2: zz/xx/yy/spin_dot/energy/dimer; "
            "ANNNI: zz/x/energy/sf/sf_pi/sf_pi_over_2."
        ),
    )
    parser.add_argument(
        "--distance",
        type=int,
        nargs="+",
        default=[1],
        help="Distance d values for distance-resolved plots such as zz/xs/xx/yy/spin_dot.",
    )
    parser.add_argument(
        "--num_h_values",
        "--num_points",
        dest="num_h_values",
        type=int,
        default=None,
        help=(
            "Number of saved parameter values to plot from each prediction file. "
            "If smaller than the saved count, values are selected by evenly spaced "
            "indices over the full parameter range. Default: use all saved values."
        ),
    )
    parser.add_argument(
        "--exact_cache_path",
        type=str,
        default="",
        help=(
            "Optional exact-value cache .npz from DshadowGPT/eval_exact_cache. "
            "When set, exact curves are loaded from this cache and plotted."
        ),
    )
    parser.add_argument(
        "--output_c",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="Plot zz_mean_from_c instead of normal zz_mean. Supports only d=1.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Directory where the figure is saved.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional full output path. Overrides --output_dir and the generated filename.",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Output figure format.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    return parser.parse_args()


def is_valid_data(data):
    if data is None:
        return False
    if isinstance(data, np.ndarray) and data.shape == () and data.item() is None:
        return False
    return True


def safe_get(data, key):
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == () and value.item() is None:
        return None
    return value


def safe_get_any(data, keys):
    for key in keys:
        value = safe_get(data, key)
        if value is not None:
            return value
    return None


def x_label_for_kind(kind):
    if kind == "J1J2":
        return r"$J_2$"
    if kind == "ANNNI":
        return r"Transverse field $g$"
    return r"$g$"


def model_title(kind):
    if kind == "J1J2":
        return "J1-J2"
    if kind == "ANNNI":
        return "ANNNI"
    return "TFI"


def make_labels(files, labels):
    if labels is not None:
        if len(labels) != len(files):
            raise ValueError("Number of labels must match number of files.")
        return labels
    return [os.path.basename(path).replace(".npz", "") for path in files]


def sanitize_filename_part(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "plot"


def make_save_path(args, labels):
    if args.output is not None:
        return args.output

    label_part = "_".join(sanitize_filename_part(label) for label in labels)
    if args.output_c:
        plot_part = "zz_from_c_d1"
    elif args.plot_type in {"zz", "xs", "xx", "yy", "spin_dot"}:
        distance_part = "_".join(str(d) for d in normalize_requested_distances(args.distance))
        plot_part = f"{args.plot_type}_d{distance_part}"
    else:
        plot_part = args.plot_type

    count_part = f"_n{args.num_h_values}" if args.num_h_values is not None else ""
    filename = f"comparisons_{plot_part}{count_part}_{label_part}.{args.save_format}"
    return os.path.join(args.output_dir, filename)


def validate_distance(distance, array, name):
    if distance < 1:
        raise ValueError("--distance must be >= 1.")
    if array is not None and distance > array.shape[0]:
        raise ValueError(f"Requested distance d={distance}, but {name} only has {array.shape[0]} distances.")
    return distance - 1


def normalize_requested_distances(distance_arg):
    if distance_arg is None:
        return None
    if isinstance(distance_arg, (list, tuple, np.ndarray)):
        raw_distances = distance_arg
    else:
        raw_distances = [distance_arg]

    distances = []
    for distance in raw_distances:
        value = int(distance)
        if value < 1:
            raise ValueError("--distance must be >= 1.")
        if value not in distances:
            distances.append(value)
    if not distances:
        raise ValueError("--distance cannot be empty.")
    return distances


def format_distances(distances):
    return ",".join(str(distance) for distance in distances)


def normalize_d_values(d_values, row_count):
    if d_values is None:
        return np.arange(1, row_count + 1, dtype=int)
    values = np.asarray(d_values, dtype=int).reshape(-1)
    if values.shape[0] < row_count:
        raise ValueError(f"d_values has {values.shape[0]} entries, but data has {row_count} rows.")
    return values[:row_count]


def find_distance_index(distance, array, name, d_values=None):
    if distance < 1:
        raise ValueError("--distance must be >= 1.")
    if not is_valid_data(array):
        raise ValueError(f"{name} is missing.")
    values = normalize_d_values(d_values, array.shape[0])
    matches = np.where(values == distance)[0]
    if matches.size == 0:
        raise ValueError(f"Requested distance d={distance}, but {name} has distances {values.tolist()}.")
    return int(matches[0])


def maybe_find_distance_index(distance, array, d_values=None):
    if not is_valid_data(array):
        return None
    values = normalize_d_values(d_values, array.shape[0])
    matches = np.where(values == distance)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def evenly_spaced_indices(total_count, selected_count):
    if selected_count is None:
        return None
    selected_count = int(selected_count)
    if selected_count <= 0:
        raise ValueError("--num_h_values must be positive.")
    if total_count <= 0:
        raise ValueError("Cannot select h values from an empty x-axis.")
    if selected_count > total_count:
        raise ValueError(
            f"--num_h_values={selected_count} exceeds the saved value count {total_count}."
        )
    if selected_count == total_count:
        return np.arange(total_count, dtype=int)
    return np.rint(np.linspace(0, total_count - 1, selected_count)).astype(int)


def subsample_prediction_values(pred, selected_count, label):
    if selected_count is None:
        return pred

    x = np.asarray(pred["x"])
    x_flat = x.reshape(-1)
    indices = evenly_spaced_indices(x_flat.shape[0], selected_count)
    if indices is None:
        return pred

    full_count = x_flat.shape[0]
    pred["x"] = x_flat[indices].reshape(-1)
    skipped_keys = {"kind", "x", "d_values"}
    for key, value in list(pred.items()):
        if key in skipped_keys or not is_valid_data(value):
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[-1] == full_count:
            pred[key] = np.take(array, indices, axis=-1)
        elif array.ndim >= 1 and array.shape[0] == full_count:
            pred[key] = np.take(array, indices, axis=0)

    print(
        f"{label}: plotting {indices.shape[0]} of {full_count} saved parameter values "
        f"(indices {indices[0]}..{indices[-1]})."
    )
    return pred


def resolve_exact_cache_path(path):
    if not path:
        return None
    candidates = [path]
    if not os.path.isabs(path):
        candidates.extend(
            [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), path),
                os.path.join(DEFAULT_EXACT_CACHE_DIR, path),
            ]
        )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Exact cache file not found: {path}")


def load_true_data(prediction_file, exact_cache_path, kind):
    true_data = np.load(prediction_file, allow_pickle=True)

    if kind == "J1J2":
        x_true = safe_get_any(true_data, ["J2_values_dense", "J2_values"])
        exact = {
            "energy": safe_get_any(true_data, ["exact_energy", "energy"]),
            "zz": safe_get_any(true_data, ["exact_corr_ZZ", "correlations_ZZ"]),
            "xx": safe_get_any(true_data, ["exact_corr_XX", "correlations_XX"]),
            "yy": safe_get_any(true_data, ["exact_corr_YY", "correlations_YY"]),
            "spin_dot": safe_get_any(true_data, ["exact_corr_spin_dot", "correlations_spin_dot"]),
            "dimer": safe_get_any(true_data, ["exact_dimer_proxy", "dimer_proxy"]),
        }
    elif kind == "ANNNI":
        x_true = safe_get_any(true_data, ["h_values_dense", "h_values", "h_values_true"])
        exact = {
            "energy": safe_get_any(true_data, ["exact_energy", "energy"]),
            "zz": safe_get_any(true_data, ["exact_ZZ_curves", "ZZ_curves"]),
            "x": safe_get_any(true_data, ["exact_X_magnetization", "X_magnetization"]),
            "sf_pi": safe_get_any(true_data, ["exact_structure_factor_pi", "structure_factor_pi"]),
            "sf_pi_over_2": safe_get_any(
                true_data,
                ["exact_structure_factor_pi_over_2", "structure_factor_pi_over_2"],
            ),
        }
    else:
        x_true = safe_get_any(true_data, ["h_values_true", "h_values"])
        exact = {
            "energy": safe_get_any(true_data, ["Energy_true", "Energy"]),
            "zz": safe_get_any(true_data, ["ZZ_curves"]),
            "xs": safe_get_any(true_data, ["Xs_curves"]),
        }

    exact_d_values = safe_get_any(true_data, ["d_values", "distances"])

    resolved_cache = resolve_exact_cache_path(exact_cache_path)
    if resolved_cache is not None:
        cache_data = np.load(resolved_cache, allow_pickle=True)
        if kind == "J1J2":
            x_true = safe_get_any(cache_data, ["J2_values", "J2_values_dense"])
            exact.update(
                {
                    "energy": safe_get_any(cache_data, ["energy", "exact_energy"]),
                    "zz": safe_get_any(cache_data, ["correlations_ZZ", "exact_corr_ZZ"]),
                    "xx": safe_get_any(cache_data, ["correlations_XX", "exact_corr_XX"]),
                    "yy": safe_get_any(cache_data, ["correlations_YY", "exact_corr_YY"]),
                    "spin_dot": safe_get_any(cache_data, ["correlations_spin_dot", "exact_corr_spin_dot"]),
                    "dimer": safe_get_any(cache_data, ["dimer_proxy", "exact_dimer_proxy"]),
                }
            )
        elif kind == "ANNNI":
            x_true = safe_get_any(cache_data, ["h_values", "h_values_dense"])
            exact.update(
                {
                    "energy": safe_get_any(cache_data, ["energy", "exact_energy"]),
                    "zz": safe_get_any(cache_data, ["ZZ_curves", "exact_ZZ_curves"]),
                    "x": safe_get_any(cache_data, ["X_magnetization", "exact_X_magnetization"]),
                    "sf_pi": safe_get_any(cache_data, ["structure_factor_pi", "exact_structure_factor_pi"]),
                    "sf_pi_over_2": safe_get_any(
                        cache_data,
                        ["structure_factor_pi_over_2", "exact_structure_factor_pi_over_2"],
                    ),
                }
            )
        else:
            x_true = safe_get_any(cache_data, ["h_values", "h_values_true"])
            exact.update(
                {
                    "zz": safe_get_any(cache_data, ["ZZ_curves"]),
                    "xs": safe_get_any(cache_data, ["Xs_curves"]),
                    "energy": safe_get_any(cache_data, ["Energy", "Energy_true"]),
                }
            )
        exact_d_values = safe_get_any(cache_data, ["d_values", "distances"])
        if x_true is None:
            raise ValueError(f"{resolved_cache} does not contain a recognized x-axis value array.")

    return {
        "kind": kind,
        "x_true": x_true,
        "exact": exact,
        "d_values": exact_d_values,
    }


def load_predictions(files, labels, args, kind):
    predictions = {}
    for file_path, label in zip(files, labels):
        data = np.load(file_path, allow_pickle=True)
        pred = {"kind": kind}
        pred["x"] = safe_get_any(data, ["hs_gpt", "J2s", "hs"])
        if pred["x"] is None:
            raise ValueError(f"{file_path} does not contain hs_gpt, J2s, or hs.")

        d_values = safe_get(data, "d_values")
        if d_values is not None:
            pred["d_values"] = np.asarray(d_values, dtype=int).reshape(-1)

        if args.output_c:
            if kind != "TFI":
                raise ValueError("--output_c is only supported for TFI output files.")
            pred["zz_mean_from_c"] = data["zz_mean_from_c"]
        else:
            if kind == "J1J2":
                pred["zz_mean"] = safe_get_any(data, ["corr_ZZ_mean", "zz_mean"])
                pred["xx_mean"] = safe_get(data, "corr_XX_mean")
                pred["yy_mean"] = safe_get(data, "corr_YY_mean")
                pred["spin_dot_mean"] = safe_get(data, "corr_spin_dot_mean")
                pred["dimer_mean"] = safe_get(data, "dimer_proxy_mean")
                pred["energy_mean"] = safe_get(data, "energy_mean")
            elif kind == "ANNNI":
                pred["zz_mean"] = safe_get(data, "zz_mean")
                pred["x_mean"] = safe_get(data, "x_mean")
                pred["sf_pi_mean"] = safe_get(data, "structure_factor_pi_mean")
                pred["sf_pi_over_2_mean"] = safe_get(data, "structure_factor_pi_over_2_mean")
                pred["energy_mean"] = safe_get(data, "energy_mean")
            else:
                pred["zz_mean"] = safe_get(data, "zz_mean")
                pred["xs_mean"] = safe_get(data, "xs_mean")
                pred["energy_mean"] = safe_get(data, "energy_mean")

        pred = subsample_prediction_values(pred, args.num_h_values, label)
        predictions[label] = pred
    return predictions


def build_style(labels):
    colors = {
        label: MODEL_COLORS[i % len(MODEL_COLORS)]
        for i, label in enumerate(labels)
    }
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    return colors, markers


def exact_color_for_distance(distance, distances):
    sorted_distances = sorted(set(int(d) for d in distances))
    max_gray = 0.0
    if len(sorted_distances) == 1:
        gray_value = max_gray
    else:
        rank = sorted_distances.index(int(distance))
        gray_value = 0.62 - (0.62 - max_gray) * rank / (len(sorted_distances) - 1)
    return (gray_value, gray_value, gray_value, 1.0)


def select_plot_distances(distance, array, name, d_values=None):
    if distance is None:
        return [int(d) for d in normalize_d_values(d_values, array.shape[0])]
    distances = normalize_requested_distances(distance)
    for d in distances:
        find_distance_index(d, array, name, d_values)
    return distances


def set_x_axis(ax, kind, x_true, predictions):
    ax.set_xlabel(x_label_for_kind(kind), fontsize=10, labelpad=3)
    xs = []
    if is_valid_data(x_true):
        xs.append(np.asarray(x_true, dtype=float).reshape(-1))
    for pred in predictions.values():
        if is_valid_data(pred.get("x")):
            xs.append(np.asarray(pred["x"], dtype=float).reshape(-1))
    if not xs:
        return
    merged = np.concatenate(xs)
    if merged.size == 0:
        return
    left = float(np.nanmin(merged))
    right = float(np.nanmax(merged))
    if np.isfinite(left) and np.isfinite(right):
        if left == right:
            pad = 0.08 if left == 0 else abs(left) * 0.05
        else:
            pad = 0.05 * (right - left)
        ax.set_xlim(left - pad, right + pad)


def plot_distance_property(
    ax,
    x_true,
    exact_array,
    exact_d_values,
    predictions,
    colors,
    markers,
    *,
    pred_key,
    ylabel,
    title,
    kind,
    distance=None,
    exact_prefix="Exact",
):
    first_pred = next(iter(predictions.values()))
    if first_pred.get(pred_key) is None:
        raise ValueError(f"Prediction files do not contain {pred_key}.")
    distances = select_plot_distances(distance, first_pred[pred_key], pred_key, first_pred.get("d_values"))
    for d in distances:
        if is_valid_data(exact_array):
            find_distance_index(d, exact_array, title, exact_d_values)
    multiple_distances = len(distances) > 1 or distance is None

    for d_idx, d in enumerate(distances):
        exact_idx = maybe_find_distance_index(int(d), exact_array, exact_d_values)
        if exact_idx is not None and x_true is not None:
            ax.plot(
                x_true,
                exact_array[exact_idx],
                color=exact_color_for_distance(d, distances),
                linestyle="-",
                linewidth=1.0,
                alpha=1.0,
                zorder=3,
                label=f"{exact_prefix} (r={int(d)})" if multiple_distances else exact_prefix,
            )

        for model_idx, (label, pred) in enumerate(predictions.items()):
            if pred.get(pred_key) is None:
                raise ValueError(f"{label} does not contain {pred_key}.")
            pred_idx = find_distance_index(int(d), pred[pred_key], pred_key, pred.get("d_values"))
            # print("plot pred_idx :", pred_idx)
            series_label = f"{label} (r={int(d)})" if multiple_distances else label
            ax.scatter(
                pred["x"],
                pred[pred_key][pred_idx],
                color=colors[label],
                marker=markers[pred_idx % len(markers)],
                s=16,
                alpha=1.0,
                edgecolors=SURFACE,
                linewidths=0.4,
                zorder=4,
                label=series_label,
            )

    if multiple_distances:
        final_title = f"{title} (r={format_distances(distances)})"
    else:
        final_title = f"{title} (r={distances[0]})"
    ax.set_ylabel(ylabel, labelpad=4.5, fontsize=10)
    # ax.set_title(final_title)
    ax.grid(False)
    # ax.set_ylim(-0.25, 1)
    set_x_axis(ax, kind, x_true, predictions)


def plot_zz(ax, h_true, zz_true, exact_d_values, predictions, colors, markers, distance=None, kind="TFI"):
    plot_distance_property(
        ax,
        h_true,
        zz_true,
        exact_d_values,
        predictions,
        colors,
        markers,
        pred_key="zz_mean",
        ylabel=r"$\langle Z_i Z_{i+r} \rangle$",
        title="Two-point ZZ correlation",
        kind=kind,
        distance=distance,
    )


def plot_xs(ax, h_true, xs_true, exact_d_values, predictions, colors, markers, distance=None, kind="TFI"):
    first_pred = next(iter(predictions.values()))
    distances = select_plot_distances(distance, first_pred["xs_mean"], "xs_mean", first_pred.get("d_values"))
    for d in distances:
        if is_valid_data(xs_true):
            find_distance_index(d, xs_true, "Xs_curves", exact_d_values)
    multiple_distances = len(distances) > 1 or distance is None

    for d_idx, d in enumerate(distances):
        exact_idx = maybe_find_distance_index(int(d), xs_true, exact_d_values)
        if exact_idx is not None and h_true is not None:
            ax.plot(
                h_true,
                xs_true[exact_idx],
                color=exact_color_for_distance(d, distances),
                linestyle="-",
                linewidth=1.0,
                alpha=1.0,
                zorder=3,
                label=f"Exact (l={int(d)})" if multiple_distances else "Exact",
            )

        for model_idx, (label, pred) in enumerate(predictions.items()):
            pred_idx = find_distance_index(int(d), pred["xs_mean"], "xs_mean", pred.get("d_values"))
            series_label = f"{label} (l={int(d)})" if multiple_distances else label
            ax.scatter(
                pred["x"],
                pred["xs_mean"][pred_idx],
                color=colors[label],
                marker=markers[pred_idx % len(markers)],
                s=10,
                alpha=1.0,
                edgecolors=SURFACE,
                linewidths=0.4,
                zorder=4,
                label=series_label,
            )

    ylabel = r"$\langle X\mathrm{-string} \rangle$"
    title = (
        f"String order parameters (l={format_distances(distances)})"
        if multiple_distances
        else f"String order parameter (l={distances[0]})"
    )
    ax.set_ylabel(ylabel, labelpad=1.5)
    # ax.set_title(title)
    ax.grid(True, alpha=0.3)
    set_x_axis(ax, kind, h_true, predictions)


def plot_scalar_property(ax, x_true, exact_values, predictions, colors, *, pred_key, ylabel, title, kind):
    if is_valid_data(exact_values) and x_true is not None:
        ax.plot(x_true, exact_values, color=INK, linewidth=1.0, label="Exact", zorder=3)

    for label, pred in predictions.items():
        if pred.get(pred_key) is None:
            raise ValueError(f"{label} does not contain {pred_key}.")
        ax.scatter(
            pred["x"],
            pred[pred_key],
            color=colors[label],
            s=10,
            alpha=1.0,
            edgecolors=SURFACE,
            linewidths=0.4,
            zorder=4,
            label=label,
        )

    ax.set_ylabel(ylabel, labelpad=1.5)
    # ax.set_title(title)
    ax.grid(True, alpha=0.3)
    set_x_axis(ax, kind, x_true, predictions)


def plot_energy(ax, h_true, energy_true, predictions, colors, kind="TFI"):
    plot_scalar_property(
        ax,
        h_true,
        energy_true,
        predictions,
        colors,
        pred_key="energy_mean",
        ylabel=r"$E_0$",
        title="Ground state energy",
        kind=kind,
    )


def plot_structure_factor(ax, x_true, true_context, predictions, colors, markers, kind="ANNNI"):
    exact_pi = true_context.get("sf_pi")
    exact_pi_over_2 = true_context.get("sf_pi_over_2")
    if is_valid_data(exact_pi) and x_true is not None:
        ax.plot(x_true, exact_pi, color=INK, linewidth=1.0, label=r"Exact $S_Z(\pi)$", zorder=3)
    if is_valid_data(exact_pi_over_2) and x_true is not None:
        ax.plot(
            x_true,
            exact_pi_over_2,
            color=INK2,
            linestyle="--",
            linewidth=1.0,
            alpha=1.0,
            zorder=3,
            label=r"Exact $S_Z(\pi/2)$",
        )

    for idx, (label, pred) in enumerate(predictions.items()):
        if pred.get("sf_pi_mean") is not None:
            ax.scatter(
                pred["x"],
                pred["sf_pi_mean"] * 0.975,
                color=colors[label],
                marker=markers[0],
                s=10,
                alpha=1.0,
                edgecolors=SURFACE,
                linewidths=0.4,
                zorder=4,
                label=rf"{label} $S_Z(\pi)$",
            )
        if pred.get("sf_pi_over_2_mean") is not None:
            ax.scatter(
                pred["x"],
                pred["sf_pi_over_2_mean"],
                color=colors[label],
                marker=markers[1],
                s=10,
                alpha=1.0,
                edgecolors=SURFACE,
                linewidths=0.4,
                zorder=4,
                label=rf"{label} $S_Z(\pi/2)$",
            )

    ax.set_ylabel(r"$S_Z(q)$", labelpad=1.5)
    # ax.set_title("Z structure factor")
    ax.grid(True, alpha=0.3)
    set_x_axis(ax, kind, x_true, predictions)


def plot_zz_from_c(ax, h_true, zz_true, exact_d_values, predictions, colors):
    exact_idx = maybe_find_distance_index(1, zz_true, exact_d_values)
    if exact_idx is not None and h_true is not None:
        ax.plot(h_true, zz_true[exact_idx], color=INK, linewidth=1.0, label="Exact", zorder=3)

    for label, pred in predictions.items():
        ax.scatter(
            pred["x"],
            pred["zz_mean_from_c"],
            color=colors[label],
            marker="o",
            s=10,
            alpha=1.0,
            edgecolors=SURFACE,
            linewidths=0.4,
            zorder=4,
            label=label,
        )

    ax.set_xlabel(r"External field $h$")
    ax.set_ylabel(r"$\langle Z_i Z_{i+1} \rangle$", labelpad=1.5)
    ax.set_title("Two-point correlation from c (d=1)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 1)


def add_deduped_legend(fig, axes):
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    handles = []
    labels = []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)

    unique = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper right",
            frameon=False,
            fontsize=6,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.6,
        )



def add_per_axis_legends(axes, fontsize=6):
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_") and label not in unique:
                unique[label] = handle
        if unique:
            ax.legend(
                unique.values(),
                unique.keys(),
                loc="best",
                fontsize=fontsize,
                frameon=False,
                handlelength=1.6,
                handletextpad=0.5,
                borderaxespad=0.3,
            )

def main():
    global plt
    args = parse_args()
    plt = import_matplotlib()
    if args.output_c and args.plot_type not in {"all", "zz"}:
        raise ValueError("--output_c only supports zz plotting.")
    if args.output_c and normalize_requested_distances(args.distance) != [1]:
        raise ValueError("--output_c only supports --distance 1.")

    kind = args.predict_model
    true_context = load_true_data(args.files[0], args.exact_cache_path, kind)

    labels = make_labels(args.files, args.labels)
    predictions = load_predictions(args.files, labels, args, kind)

    x_true = true_context["x_true"]
    exact = true_context["exact"]
    exact_d_values = true_context["d_values"]

    apply_figure_style()
    colors, markers = build_style(labels)

    if args.output_c:
        if kind != "TFI":
            raise ValueError("--output_c only supports TFI output files.")
        fig, ax = plt.subplots(1, 1, figsize=(3.54, 2.7), facecolor=SURFACE)
        plot_zz_from_c(ax, x_true, exact.get("zz"), exact_d_values, predictions, colors)
        axes = np.asarray([ax])
    elif args.plot_type == "all":
        if kind == "J1J2":
            fig, axes = plt.subplots(2, 2, figsize=(7.087, 5.0), facecolor=SURFACE)
            axes = axes.reshape(2, 2)
            plot_zz(axes[0, 0], x_true, exact.get("zz"), exact_d_values, predictions, colors, markers, kind=kind)
            plot_distance_property(
                axes[0, 1],
                x_true,
                exact.get("spin_dot"),
                exact_d_values,
                predictions,
                colors,
                markers,
                pred_key="spin_dot_mean",
                ylabel=r"$\langle \sigma_i\cdot\sigma_{i+d}\rangle$",
                title="Spin-dot correlation",
                kind=kind,
            )
            plot_energy(axes[1, 0], x_true, exact.get("energy"), predictions, colors, kind=kind)
            plot_scalar_property(
                axes[1, 1],
                x_true,
                exact.get("dimer"),
                predictions,
                colors,
                pred_key="dimer_mean",
                ylabel=r"$C_1-C_2$",
                title="Dimer/frustration proxy",
                kind=kind,
            )
        elif kind == "ANNNI":
            fig, axes = plt.subplots(2, 2, figsize=(7.087, 5.0), facecolor=SURFACE)
            axes = axes.reshape(2, 2)
            plot_zz(axes[0, 0], x_true, exact.get("zz"), exact_d_values, predictions, colors, markers, kind=kind)
            plot_scalar_property(
                axes[0, 1],
                x_true,
                exact.get("x"),
                predictions,
                colors,
                pred_key="x_mean",
                ylabel=r"$\langle X\rangle$",
                title="X magnetization",
                kind=kind,
            )
            plot_energy(axes[1, 0], x_true, exact.get("energy"), predictions, colors, kind=kind)
            plot_structure_factor(axes[1, 1], x_true, exact, predictions, colors, markers, kind=kind)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(7.087, 2.35), facecolor=SURFACE)
            plot_zz(axes[0], x_true, exact.get("zz"), exact_d_values, predictions, colors, markers, distance=None, kind=kind)
            plot_xs(axes[1], x_true, exact.get("xs"), exact_d_values, predictions, colors, markers, distance=None, kind=kind)
            plot_energy(axes[2], x_true, exact.get("energy"), predictions, colors, kind=kind)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(3.54, 2.7), facecolor=SURFACE)
        if args.plot_type == "zz":
            plot_zz(ax, x_true, exact.get("zz"), exact_d_values, predictions, colors, markers, distance=args.distance, kind=kind)
        elif args.plot_type == "xs":
            if kind != "TFI":
                raise ValueError("--plot_type xs is only available for TFI outputs.")
            plot_xs(ax, x_true, exact.get("xs"), exact_d_values, predictions, colors, markers, distance=args.distance, kind=kind)
        elif args.plot_type == "xx":
            if kind != "J1J2":
                raise ValueError("--plot_type xx is only available for J1J2 outputs.")
            plot_distance_property(
                ax,
                x_true,
                exact.get("xx"),
                exact_d_values,
                predictions,
                colors,
                markers,
                pred_key="xx_mean",
                ylabel=r"$\langle X_i X_{i+d}\rangle$",
                title="XX correlation",
                kind=kind,
                distance=args.distance,
            )
        elif args.plot_type == "yy":
            if kind != "J1J2":
                raise ValueError("--plot_type yy is only available for J1J2 outputs.")
            plot_distance_property(
                ax,
                x_true,
                exact.get("yy"),
                exact_d_values,
                predictions,
                colors,
                markers,
                pred_key="yy_mean",
                ylabel=r"$\langle Y_i Y_{i+d}\rangle$",
                title="YY correlation",
                kind=kind,
                distance=args.distance,
            )
        elif args.plot_type == "spin_dot":
            if kind != "J1J2":
                raise ValueError("--plot_type spin_dot is only available for J1J2 outputs.")
            plot_distance_property(
                ax,
                x_true,
                exact.get("spin_dot"),
                exact_d_values,
                predictions,
                colors,
                markers,
                pred_key="spin_dot_mean",
                ylabel=r"$\langle \sigma_i\cdot\sigma_{i+d}\rangle$",
                title="Spin-dot correlation",
                kind=kind,
                distance=args.distance,
            )
        elif args.plot_type == "energy":
            plot_energy(ax, x_true, exact.get("energy"), predictions, colors, kind=kind)
        elif args.plot_type == "dimer":
            if kind != "J1J2":
                raise ValueError("--plot_type dimer is only available for J1J2 outputs.")
            plot_scalar_property(
                ax,
                x_true,
                exact.get("dimer"),
                predictions,
                colors,
                pred_key="dimer_mean",
                ylabel=r"$C_1-C_2$",
                title="Dimer/frustration proxy",
                kind=kind,
            )
        elif args.plot_type == "x":
            if kind != "ANNNI":
                raise ValueError("--plot_type x is only available for ANNNI outputs.")
            plot_scalar_property(
                ax,
                x_true,
                exact.get("x"),
                predictions,
                colors,
                pred_key="x_mean",
                ylabel=r"$\langle X\rangle$",
                title="X magnetization",
                kind=kind,
            )
        elif args.plot_type == "sf":
            if kind != "ANNNI":
                raise ValueError("--plot_type sf is only available for ANNNI outputs.")
            plot_structure_factor(ax, x_true, exact, predictions, colors, markers, kind=kind)
        elif args.plot_type == "sf_pi":
            if kind != "ANNNI":
                raise ValueError("--plot_type sf_pi is only available for ANNNI outputs.")
            plot_scalar_property(
                ax,
                x_true,
                exact.get("sf_pi"),
                predictions,
                colors,
                pred_key="sf_pi_mean",
                ylabel=r"$S_Z(\pi)$",
                title=r"Z structure factor $q=\pi$",
                kind=kind,
            )
        elif args.plot_type == "sf_pi_over_2":
            if kind != "ANNNI":
                raise ValueError("--plot_type sf_pi_over_2 is only available for ANNNI outputs.")
            plot_scalar_property(
                ax,
                x_true,
                exact.get("sf_pi_over_2"),
                predictions,
                colors,
                pred_key="sf_pi_over_2_mean",
                ylabel=r"$S_Z(\pi/2)$",
                title=r"Z structure factor $q=\pi/2$",
                kind=kind,
            )
        axes = np.asarray([ax])

    for axis in np.asarray(axes).flat:
        style_axis(axis)

    if args.plot_type == "all":
        add_per_axis_legends(axes)
        fig.tight_layout(pad=0.5)
    else:
        add_deduped_legend(fig, axes)
        fig.tight_layout(pad=0.5)

    save_path = make_save_path(args, labels)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight", facecolor=SURFACE)
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
