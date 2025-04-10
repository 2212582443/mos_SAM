from collections import OrderedDict
from statistics import mean
from typing import Any, List
from glob import glob
import torch, re, os, json
from scipy.stats import shapiro, ttest_rel, pearsonr, t
from datetime import datetime
from json import JSONEncoder

ExperimentNamePattern = re.compile(
    "(?P<name>.+?)(?P<tag>\[p\])?-(?P<dim>[23]d)-(?P<dataset>[^\[]+)(?P<extra>\[.+\])?$",
    re.IGNORECASE,
)

LATEX_DOC_ROOT = ".checkpoint/summary/thesis"
MICCAI_LATEX_DOC_ROOT = ".checkpoint/summary/miccai"

ALL_EXPERIMENT_DISPLAY_ORDER = [
    ("resunet", "ResUnet"),
    ("resunet++", "ResUnet++"),
    ("segnet", "SegNet"),
    ("unet[s]", "Unet[s]"),
    ("unet", "Unet"),
    ("unetr", "UNETR"),
    ("sam", "SAM"),
    ("effsams", "EffSAMs"),
    ("effsamti", "EffSAMti"),
    ("txtsam", "Our"),
    ("txtsam[p]", "Our[p]"),
    ("txtsam[s]", "Our[s]"),
    ("txtsam[ps]", "Our[ps]"),
]
LATEX_DISPLAY_ORDER = [ALL_EXPERIMENT_DISPLAY_ORDER[i] for i in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]]


class ExperimentName(JSONEncoder):
    def __init__(self, experiment_name) -> None:
        global ExperimentNamePattern
        m = ExperimentNamePattern.search(experiment_name)
        name, tag, dim, dataset, extra = (
            m.group("name"),
            m.group("tag"),
            m.group("dim"),
            m.group("dataset"),
            m.group("extra"),
        )
        if tag is None:
            tag = ""
        self.name = name + tag
        if len(tag) > 2:
            tag = tag[1:-1]
            tag = set(tag.split(","))
        self.tag = tag
        self.dim = 2 if dim == "2d" else 3
        self.aux_dataset = dataset != "none"

        if extra is None:
            extra = ""
        self.extra = extra.strip()

    def __str__(self) -> str:
        return self.__dict__.__str__()

    def default(self, o):
        return o.__dict__


class ExperimentResult(JSONEncoder):
    def __init__(self) -> None:
        self.settings: list[dict[str, dict[str, torch.Tensor]]] = [
            dict({"2d": {}, "3d": {}, "2daux": {}, "3daux": {}}) for _ in range(5)
        ]
        self.zeroshot_settings: list[dict[str, dict[str, torch.Tensor]]] = [
            dict({"2d": {}, "3d": {}, "2daux": {}, "3daux": {}}) for _ in range(5)
        ]

    def get_result(self, setting_id: int, name: ExperimentName) -> dict:
        name = f"{name.dim}daux" if name.aux_dataset else f"{name.dim}d"
        return self.settings[setting_id][name]

    def get_zeroshot_result(self, setting_id: int, name: ExperimentName) -> dict:
        # assert name.aux_dataset

        name = f"{name.dim}daux" if name.aux_dataset else f"{name.dim}d"
        return self.zeroshot_settings[setting_id][name]

    def __str__(self) -> str:
        return self.__dict__.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def default(self, o):
        return o.__dict__


def find_best_epoch(experiment_root: str) -> str:
    best_dice, best_epoch = 0, ""

    metric_file_list = glob("epoch-*/valid-metrics.pt", root_dir=experiment_root)
    metric_file_list.sort(reverse=True)
    # metric_file_list = metric_file_list[:20]

    for metric in metric_file_list:
        epoch = f"{experiment_root}/{metric[:-17]}/test-metrics.pt"
        if not os.path.exists(epoch):
            continue
        metric = torch.load(f"{experiment_root}/{metric}")
        dice: torch.Tensor = metric["dice"]
        dice = dice.mean().item()
        if dice > best_dice:
            best_dice = dice
            best_epoch = epoch
    if len(best_epoch) == 0:  # 早期少了实验没有valid-metrics.pt文件，取最后一个结果
        epoch = f"{experiment_root}/latest/test-metrics.pt"
        if os.path.exists(epoch):
            best_epoch = epoch

    if len(best_epoch) == 0:
        print("best metric NOT found!", experiment_root)
    else:
        print("best metric:", best_dice, best_epoch)
    return best_epoch


def mark_best(all_result: dict[str, ExperimentResult]):
    for metric in ["dice", "iou", "msd", "hd"]:
        lower_better = metric == "msd" or metric == "hd"
        for setting_id in range(5):
            for experiment_key in ["2d", "3d", "2daux", "3daux"]:
                best_score, best_experiment = None, None
                for _key, experiment in all_result.items():
                    current = experiment.settings[setting_id][experiment_key]
                    if metric not in current:
                        continue
                    current = current[metric]
                    current_score = current.mean().item()
                    if best_experiment is None:
                        best_score = current_score
                        best_experiment = current
                    elif lower_better and current_score < best_score:
                        best_score = current_score
                        best_experiment = current
                    elif (not lower_better) and current_score > best_score:
                        best_score = current_score
                        best_experiment = current
                if best_experiment is not None:
                    setattr(best_experiment, "best", True)

    for metric in ["dice", "iou", "msd", "hd"]:
        lower_better = metric == "msd" or metric == "hd"
        for setting_id in range(5):
            for experiment_key in ["2daux", "3daux"]:
                best_score, best_experiment = None, None
                for _key, experiment in all_result.items():
                    current = experiment.zeroshot_settings[setting_id][experiment_key]
                    if metric not in current:
                        continue
                    current = current[metric]
                    current_score = current.mean().item()
                    if best_experiment is None:
                        best_score = current_score
                        best_experiment = current
                    elif lower_better and current_score < best_score:
                        best_score = current_score
                        best_experiment = current
                    elif (not lower_better) and current_score > best_score:
                        best_score = current_score
                        best_experiment = current
                if best_experiment is not None:
                    setattr(best_experiment, "best", True)


def plot_table_detail(key: str, all_result: dict[str, ExperimentResult]) -> str:
    table = f"""<table class="pure-table metric-table">
    <caption>metric:{key}</caption><tbody>
    <tr>
    <th rowspan=3 class="bb center">{key}</th>
    """
    for i in range(5):
        table += f"<th colspan=4 class='bb center'>fold-{i}</th>"
    table += "</tr><tr>"
    for _ in range(5):
        table += f"<td colspan=2 class='bb center'>主数据</td><td colspan=2 class='bb center'>主数据+辅助数据</td>"
    table += "</tr><tr>"
    for _ in range(10):
        table += f"<td class='bb center'>2D</td><td class='bb center'>3D</td>"
    table += "</tr>"

    for name, result in all_result.items():
        table += f"<tr class='data'><td>{name}</td>"
        for setting_id in range(5):
            experiments: dict[str, dict[str, torch.Tensor]] = result.settings[setting_id]
            for experiment_key in ["2d", "3d", "2daux", "3daux"]:
                experiment = experiments[experiment_key]
                if key in experiment:
                    experiment = experiment[key]
                    m = experiment.mean().item()
                    std = experiment.std().item()
                    best = hasattr(experiment, "best")
                    if best:
                        value = (
                            f"<pre class=best>{m:>7.4f}<sub>±{std:>6.3f}</sub></pre>"
                            if key == "hd"
                            else f"<pre class=best>{m:>7.4f}<sub>±{std:>6.4f}</sub></pre>"
                        )
                    else:
                        value = (
                            f"<pre>{m:>7.4f}<sub>±{std:>6.3f}</sub></pre>"
                            if key == "hd"
                            else f"<pre>{m:>7.4f}<sub>±{std:>6.4f}</sub></pre>"
                        )

                else:
                    value = "<center>-<center>"
                table += f"<td class='d{experiment_key[:2]}'>{value}</td>"

        table += "</tr>"

    table += "</tbody></table>"

    return table


def latex_table_detail(
    key: str,
    all_result: dict[str, ExperimentResult],
    show_items=LATEX_DISPLAY_ORDER,
    table_part="1",
) -> str:
    show_items = [item for item in show_items if item[0] in all_result]
    value_table = []
    for name, display_name in show_items:
        result = all_result[name]
        for experiment_key in ["2d", "3d", "2daux", "3daux"]:
            row = []
            for setting_id in range(5):
                experiments: dict[str, dict[str, torch.Tensor]] = result.settings[setting_id]
                experiment = experiments[experiment_key]
                if key in experiment:
                    experiment = experiment[key]
                    m = experiment.mean().item()
                    std = experiment.std().item()
                    row.append(
                        {
                            0: m,
                            1: std,
                            "lower_better": key == "msd" or key == "hd",
                        }
                    )
                else:
                    row.append(None)
            value_table.append(row)
    # 标记最优的数值
    for cell_index in range(5):
        best_value, best_item = None, None
        for row in value_table:
            current = row[cell_index]
            if current is None:
                continue
            if best_item is None:
                best_value = current[0]
                best_item = current
            elif current["lower_better"] and current[0] < best_value:
                best_value = current[0]
                best_item = current
            elif (not current["lower_better"]) and current[0] > best_value:
                best_value = current[0]
                best_item = current
        if best_item is not None:
            best_item["best"] = True

    table = r"""
    \begin{table}[!htpb]
    \caption{模型交叉验证详细信息--$key$指标$table_part$}
    \label{tab:cross-validation-result-$key$-$table_part$}
    \centering
    \begin{threeparttable}[b]
        \begin{tabular}{cccccccc}
            \toprule
            \multicolumn{3}{c}{交叉验证结果[$key$]} & fold-1                & fold-2                     & fold-3                     & fold-4                     & fold-5                                                                     \\

"""
    table = table.replace("$key$", key).replace("$table_part$", table_part)
    row_index = 0
    for name, display_name in show_items:
        result = all_result[name]
        table += "\\midrule\n"
        for experiment_key in ["2d", "3d", "2daux", "3daux"]:
            if experiment_key == "2d":
                table += r"\multirow{4}*{" + display_name + r"}              & \multirow{2}*{N} & 2D" + "\n"
            elif experiment_key == "2daux":
                table += r"\cmidrule(lr){2-8}               & \multirow{2}*{Y} & 2D" + "\n"
            else:
                table += r"               &               & 3D" + "\n"

            for setting_id in range(5):
                value = value_table[row_index][setting_id]
                if value is None:
                    value = "& - "
                elif "best" in value:
                    m, std = value[0], value[1]
                    if key == "hd":
                        value = r"& $\mathbf{ " + f"{m:0.3f}" + r"_{\pm " + f"{std:0.3f}" + r"}}$"
                    else:
                        value = r"& $\mathbf{ " + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}}$"
                else:
                    m, std = value[0], value[1]
                    if key == "hd":
                        value = r"& $" + f"{m:0.3f}" + r"_{\pm " + f"{std:0.3f}" + r"}$"
                    else:
                        value = r"& $" + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}$"
                table += value
            table += "\\\\\n"
            row_index += 1

    table += r"""            \bottomrule
        \end{tabular}
    \end{threeparttable}
\end{table}"""

    # save to file
    os.makedirs(f"{LATEX_DOC_ROOT}/figures/appendex", exist_ok=True)
    with open(
        f"{LATEX_DOC_ROOT}/figures/appendex/cross-validation-result-{key}-{table_part}.tex", "w", encoding="utf-8"
    ) as f:
        f.write(table)


def plot_table_detail_zeroshot(key: str, all_result: dict[str, ExperimentResult]) -> str:
    table = f"""<table class="pure-table metric-table">
    <caption>zeroshot metric:{key}</caption><tbody>
    <tr>
    <th rowspan=2 class="bb center">{key}</th>
    """
    for i in range(5):
        table += f"<th colspan=2 class='bb center'>fold-{i}</th>"
    table += "</tr><tr>"
    for _ in range(5):
        table += f"<td class='bb center'>2D</td><td class='bb center'>3D</td>"
    table += "</tr>"

    for name, result in all_result.items():
        table += f"<tr class='data'><td>{name}</td>"
        for setting_id in range(5):
            experiments: dict[str, dict[str, torch.Tensor]] = result.zeroshot_settings[setting_id]
            for experiment_key in ["2daux", "3daux"]:
                experiment = experiments[experiment_key]
                if key in experiment:
                    experiment = experiment[key]
                    experiment = experiment[torch.isfinite(experiment)]
                    if experiment.numel() == 0:
                        value = "<center>-<center>"
                    else:
                        m = experiment.mean().item()
                        std = experiment.std().item()
                        best = hasattr(experiment, "best")
                        if best:
                            value = (
                                f"<pre class=best>{m:>7.4f}<sub>±{std:>6.3f}</sub></pre>"
                                if key == "hd"
                                else f"<pre class=best>{m:>7.4f}<sub>±{std:>6.4f}</sub></pre>"
                            )
                        else:
                            value = (
                                f"<pre>{m:>7.4f}<sub>±{std:>6.3f}</sub></pre>"
                                if key == "hd"
                                else f"<pre>{m:>7.4f}<sub>±{std:>6.4f}</sub></pre>"
                            )

                else:
                    value = "<center>-<center>"
                table += f"<td class='d{experiment_key[:2]}'>{value}</td>"

        table += "</tr>"

    table += "</tbody></table>"

    return table


def latex_table_detail_zeroshot(key: str, all_result: dict[str, ExperimentResult]) -> str:
    show_items = [item for item in all_result if item[0] in LATEX_DISPLAY_ORDER]
    value_table = []
    for name, display_name in show_items:
        result = all_result[name]
        for experiment_key in ["2daux", "3daux"]:
            row = []
            for setting_id in range(5):
                experiments: dict[str, dict[str, torch.Tensor]] = result.zeroshot_settings[setting_id]
                experiment = experiments[experiment_key]
                if key in experiment:
                    experiment = experiment[key]
                    # 排除nan和inf
                    experiment = experiment[torch.isfinite(experiment)]
                    if experiment.numel() == 0:
                        row.append(None)
                        continue
                    m = experiment.mean().item()
                    std = experiment.std().item()
                    row.append(
                        {
                            0: m,
                            1: std,
                            "lower_better": key == "msd" or key == "hd",
                        }
                    )
                else:
                    row.append(None)
            value_table.append(row)
    # 标记最优的数值
    for cell_index in range(5):
        best_value, best_item = None, None
        for row in value_table:
            current = row[cell_index]
            if current is None:
                continue
            if best_item is None:
                best_value = current[0]
                best_item = current
            elif current["lower_better"] and current[0] < best_value:
                best_value = current[0]
                best_item = current
            elif (not current["lower_better"]) and current[0] > best_value:
                best_value = current[0]
                best_item = current
        if best_item is not None:
            best_item["best"] = True

    table = r"""
    \begin{table}[!htpb]
    \caption{zeroshot交叉验证详细信息--$key$指标}
    \label{tab:zeroshot-detail-$key$}
    \centering
    \begin{threeparttable}[b]
        \begin{tabular}{ccccccc}
            \toprule
            \multicolumn{2}{c}{zeroshot-[$key$]} & fold-1                & fold-2                     & fold-3                     & fold-4                     & fold-5                                                                     \\

"""
    table = table.replace("$key$", key)
    row_index = 0
    for name, display_name in show_items:
        result = all_result[name]
        table += "\\midrule\n"
        for experiment_key in ["2daux", "3daux"]:
            if experiment_key == "2daux":
                table += r"\multirow{2}*{" + display_name + r"}    & 2D" + "\n"
            elif experiment_key == "3daux":
                table += r"        &  3D" + "\n"

            for setting_id in range(5):
                value = value_table[row_index][setting_id]
                if value is None:
                    value = "& - "
                elif "best" in value:
                    m, std = value[0], value[1]
                    if key == "hd":
                        value = r"& $\mathbf{ " + f"{m:0.3f}" + r"_{\pm " + f"{std:0.3f}" + r"}}$"
                    else:
                        value = r"& $\mathbf{ " + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}}$"
                else:
                    m, std = value[0], value[1]
                    if key == "hd":
                        value = r"& $" + f"{m:0.3f}" + r"_{\pm " + f"{std:0.3f}" + r"}$"
                    else:
                        value = r"& $" + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}$"
                table += value
            table += "\\\\\n"
            row_index += 1

    table += r"""            \bottomrule
        \end{tabular}
    \end{threeparttable}
\end{table}"""

    # save to file
    os.makedirs(f"{LATEX_DOC_ROOT}/figures/appendex", exist_ok=True)
    with open(f"{LATEX_DOC_ROOT}/figures/appendex/zeroshot-detail-{key}.tex", "w", encoding="utf-8") as f:
        f.write(table)


def plot_table_avage(all_result: dict[str, ExperimentResult]) -> str:
    table = f"""<table class="pure-table metric-table">
    <caption>avage metric</caption><tbody>
    <tr>
    <th rowspan=3 class="bb center">&nbsp;</th>
    """
    metric_list = ["dice", "iou", "msd", "hd"]
    for metric in metric_list:
        table += f"<th colspan=4 class='bb center'>{metric}</th>"
    table += "</tr><tr>"
    for _ in range(len(metric_list)):
        table += f"<td colspan=2 class='bb center'>主数据</td><td colspan=2 class='bb center'>主数据+辅助数据</td>"
    table += "</tr><tr>"
    for _ in range(2 * len(metric_list)):
        table += f"<td class='bb center'>2D</td><td class='bb center'>3D</td>"
    table += "</tr>"

    value_table = {}
    for name, result in all_result.items():
        if name not in value_table:
            value_table[name] = []
        for metric in metric_list:
            lower_better = metric == "msd" or metric == "hd"
            for experiment_key in ["2d", "3d", "2daux", "3daux"]:
                value_list = []
                for setting_id in range(5):
                    experiments: dict[str, dict[str, torch.Tensor]] = result.settings[setting_id]
                    experiment = experiments[experiment_key]
                    if metric in experiment:
                        experiment = experiment[metric]
                        value_list.append(experiment)
                if len(value_list) == 0:
                    value_table[name].append(None)
                else:
                    value_list = torch.cat(value_list)
                    value_list = value_list[torch.isfinite(value_list)]
                    if value_list.numel() == 0:
                        value_table[name].append(None)
                        continue
                    item = {
                        0: value_list.mean().item(),
                        1: value_list.std().item(),
                        "lower_better": lower_better,
                    }
                    value_table[name].append(item)
    for i in range(4 * len(metric_list)):
        best_value, best_item = None, None
        for _, value in value_table.items():
            current = value[i]
            if current is None:
                continue
            if best_item is None:
                best_value = current[0]
                best_item = current
            elif current["lower_better"] and current[0] < best_value:
                best_value = current[0]
                best_item = current
            elif (not current["lower_better"]) and current[0] > best_value:
                best_value = current[0]
                best_item = current
        if best_item is not None:
            best_item["best"] = True

    # print(value_table)
    for name in all_result.keys():
        table += f"<tr class='data'><td>{name}</td>"
        for idx, value in enumerate(value_table[name]):
            if value is None:
                value = "<center>-<center>"
            elif "best" in value:
                m, std = value[0], value[1]
                value = f"<pre class=best>{m:>7.4f}<sub>±{std:>6.4f}</sub></pre>"
            else:
                m, std = value[0], value[1]
                value = f"<pre>{m:>7.4f}<sub>±{std:>6.4f}</sub></pre>"

            cls = "d2d" if idx % 2 == 0 else "d3d"
            table += f"<td class='{cls}'>{value}</td>"

        table += "</tr>"

    table += "</tbody></table>"

    return table


def latex_table_avage(
    all_result: dict[str, ExperimentResult],
    table_id="overview",
    show_items=LATEX_DISPLAY_ORDER,
    title="基准实验结果",
    title_en="Model experiment results",
    footnote="",
    split_keys=set("txtsam"),
) -> str:
    show_items = [item for item in show_items if item[0] in all_result]
    for metric_index, metric_list in enumerate([["dice", "iou"], ["msd", "hd"]]):
        table = r"""
    \begin{table}[!htpb]
        \bicaption{$title$ （$result$）}{$title_en$ part $result$}
        \label{tab:$table_id$-result-$index$}
        \centering
        \begin{threeparttable}[b]
            \begin{tabular}{cccccc}
                \toprule
                \multirow{2}*{实验}   & 指标     & \multicolumn{2}{c}{$metric1$} & \multicolumn{2}{c}{$metric2$}                                                 \\
                \cmidrule(lr){2-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6}
                                    & 辅助数据 & N                        & Y                       & N                     & Y                     \\
        """
        table = (
            table.replace("$index$", str(metric_index))
            .replace("$result$", "\&".join(metric_list))
            .replace("$metric1$", metric_list[0])
            .replace("$metric2$", metric_list[1])
            .replace("$table_id$", table_id)
            .replace("$title$", title)
            .replace("$title_en$", title_en)
        )

        # 计算出每个格子的数据
        # experiment, 2d, [metric1, metric2,...]
        # experiment, 3d, [metric1, metric2,...]
        value_table = []
        for name, _display_name in show_items:
            result = all_result[name]
            row = []
            for metric in metric_list:
                value_statistic = {}
                lower_better = metric == "msd" or metric == "hd"
                for experiment_key in ["2d", "2daux"]:
                    value_list = []
                    for setting_id in range(5):
                        experiments: dict[str, dict[str, torch.Tensor]] = result.settings[setting_id]
                        experiment = experiments[experiment_key]
                        if metric in experiment:
                            experiment = experiment[metric]
                            value_list.append(experiment)
                    if len(value_list) == 0:
                        value_statistic[experiment_key] = None
                    else:
                        value_list = torch.cat(value_list)
                        value_list = value_list[torch.isfinite(value_list)]
                        if value_list.numel() == 0:
                            value_statistic[experiment_key] = None
                            continue
                        item = {
                            0: value_list.mean().item(),
                            1: value_list.std().item(),
                            "lower_better": lower_better,
                        }
                        value_statistic[experiment_key] = item
                row.append(value_statistic["2d"])
                row.append(value_statistic["2daux"])
            value_table.append(row)
            row = []
            for metric in metric_list:
                value_statistic = {}
                lower_better = metric == "msd" or metric == "hd"
                for experiment_key in ["3d", "3daux"]:
                    value_list = []
                    for setting_id in range(5):
                        experiments: dict[str, dict[str, torch.Tensor]] = result.settings[setting_id]
                        experiment = experiments[experiment_key]
                        if metric in experiment:
                            experiment = experiment[metric]
                            value_list.append(experiment)
                    if len(value_list) == 0:
                        value_statistic[experiment_key] = None
                    else:
                        value_list = torch.cat(value_list)
                        item = {
                            0: value_list.mean().item(),
                            1: value_list.std().item(),
                            "lower_better": lower_better,
                        }
                        value_statistic[experiment_key] = item
                row.append(value_statistic["3d"])
                row.append(value_statistic["3daux"])
            value_table.append(row)

        for cell_index in range(2 * len(metric_list)):
            best_value, best_item = None, None
            for row in value_table:
                current = row[cell_index]
                if current is None:
                    continue
                if best_item is None:
                    best_value = current[0]
                    best_item = current
                elif current["lower_better"] and current[0] < best_value:
                    best_value = current[0]
                    best_item = current
                elif (not current["lower_better"]) and current[0] > best_value:
                    best_value = current[0]
                    best_item = current
            if best_item is not None:
                best_item["best"] = True

        # print(value_table)
        row_index = 0
        for name, display_name in show_items:
            for dim in ["2D", "3D"]:
                if row_index % 2 == 0:
                    table += "\\midrule \n"
                    if name in split_keys:
                        table += "\\midrule \n"
                    table += r"\multirow{2}*{" + display_name + "}   & " + dim
                else:
                    table += " & " + dim
                for value in value_table[row_index]:
                    if value is None:
                        value = "& - "
                    elif "best" in value:
                        m, std = value[0], value[1]
                        value = r"& $\mathbf{ " + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}}$"
                    else:
                        m, std = value[0], value[1]
                        value = r"& $" + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}$"
                    table += value

                table += "\\\\ \n"
                row_index += 1

        table += r"""
                    \bottomrule
        \end{tabular}
        \begin{tablenotes}
            \item [Y/N] 代表是否使用辅助数据集进行微调
            \item [p] 代表使用预训练模型
            $footnote$
        \end{tablenotes}
    \end{threeparttable}
\end{table}
""".replace(
            "$footnote$", footnote
        )
        os.makedirs(f"{LATEX_DOC_ROOT}/figures/experiments", exist_ok=True)
        # save to file
        with open(
            f"{LATEX_DOC_ROOT}/figures/experiments/{table_id}-result-{metric_index}.tex", "w", encoding="utf-8"
        ) as f:
            f.write(table)


def miccai_latex_table_avage1(
    all_result: dict[str, ExperimentResult],
    table_id="overview",
    show_items=LATEX_DISPLAY_ORDER,
    title="基准实验结果",
    title_en="Experiment results",
    footnote="",
    split_keys=set("txtsam"),
) -> str:
    show_items = [item for item in show_items if item[0] in all_result]
    for metric_index, metric_list in enumerate([["dice", "iou", "msd", "hd"]]):
        table = r"""
    \begin{table}[!t]
        \caption{$title_en$}
        \label{tab:$table_id$-result-$index$}
        \centering
        \begin{threeparttable}[b]
            \scalebox{0.93}{
            \begin{tabular}{lccccccccc}
                \toprule
                \multicolumn{2}{l}{\multirow{2}{*}{Experiments}} & \multicolumn{2}{c}{ $Dice_\%$} & \multicolumn{2}{c}{ $IoU_\%$ } & \multicolumn{2}{c}{ $MSD_{mm}$} & \multicolumn{2}{c}{$HD_{mm}$} \\
                \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
                                    &  & N  & Y  & N & Y  & N & Y & N & Y  \\
        """
        table = (
            table.replace("$index$", str(metric_index))
            .replace("$result$", "\&".join(metric_list))
            .replace("$metric1$", metric_list[0])
            .replace("$metric2$", metric_list[1])
            .replace("$metric3$", metric_list[2])
            .replace("$metric4$", metric_list[3])
            .replace("$table_id$", table_id)
            .replace("$title$", title)
            .replace("$title_en$", title_en)
        )

        # 计算出每个格子的数据
        # experiment, 2d, [metric1, metric2,...]
        # experiment, 3d, [metric1, metric2,...]
        value_table = []
        for name, _display_name in show_items:
            result = all_result[name]
            row = []
            for metric in metric_list:
                value_statistic = {}
                lower_better = metric == "msd" or metric == "hd"
                for experiment_key in ["2d", "2daux"]:
                    value_list = []
                    for setting_id in range(5):
                        experiments: dict[str, dict[str, torch.Tensor]] = result.settings[setting_id]
                        experiment = experiments[experiment_key]
                        if metric in experiment:
                            experiment = experiment[metric]
                            value_list.append(experiment)
                    if len(value_list) == 0:
                        value_statistic[experiment_key] = None
                    else:
                        value_list = torch.cat(value_list)
                        value_list = value_list[torch.isfinite(value_list)]
                        if value_list.numel() == 0:
                            value_statistic[experiment_key] = None
                            continue
                        raw_data = value_list
                        value_list = value_list * 100 if metric in ["dice", "iou"] else value_list
                        item = {
                            0: value_list.mean().item(),
                            1: value_list.std().item(),
                            "raw_data": raw_data,
                            "lower_better": lower_better,
                        }
                        value_statistic[experiment_key] = item
                row.append(value_statistic["2d"])
                row.append(value_statistic["2daux"])
            value_table.append(row)
            row = []
            for metric in metric_list:
                value_statistic = {}
                lower_better = metric == "msd" or metric == "hd"
                for experiment_key in ["3d", "3daux"]:
                    value_list = []
                    for setting_id in range(5):
                        experiments: dict[str, dict[str, torch.Tensor]] = result.settings[setting_id]
                        experiment = experiments[experiment_key]
                        if metric in experiment:
                            experiment = experiment[metric]
                            value_list.append(experiment)
                    if len(value_list) == 0:
                        value_statistic[experiment_key] = None
                    else:
                        value_list = torch.cat(value_list)
                        raw_data = value_list
                        value_list = value_list * 100 if metric in ["dice", "iou"] else value_list
                        item = {
                            0: value_list.mean().item(),
                            1: value_list.std().item(),
                            "raw_data": raw_data,
                            "lower_better": lower_better,
                        }
                        value_statistic[experiment_key] = item
                row.append(value_statistic["3d"])
                row.append(value_statistic["3daux"])
            value_table.append(row)

        for cell_index in range(2 * len(metric_list)):
            best_value, best_item = None, None
            for row in value_table:
                cell = row[cell_index]
                if cell is None:
                    continue
                if best_item is None:
                    best_value = cell[0]
                    best_item = cell
                elif cell["lower_better"] and cell[0] < best_value:
                    best_value = cell[0]
                    best_item = cell
                elif (not cell["lower_better"]) and cell[0] > best_value:
                    best_value = cell[0]
                    best_item = cell
            if best_item is None:
                continue

            best_item["best"] = True
            # calc t-test with best_item
            best_item = best_item["raw_data"].tolist()
            for row in value_table:
                cell = row[cell_index]
                if cell is None or cell is best_item:
                    continue
                current = cell["raw_data"].tolist()
                src, target = [], []
                for a, b in zip(best_item, current):
                    if a is not None and b is not None:
                        src.append(a)
                        target.append(b)
                ttest_result = ttest_rel(src, target)
                statistic, pvalue = ttest_result.statistic, ttest_result.pvalue
                cell["pvalue"] = pvalue

        # print(value_table)
        row_index = 0
        for name, display_name in show_items:
            for dim in ["2D", "3D"]:
                if row_index % 2 == 0:
                    table += "\\midrule \n"
                    if name in split_keys:
                        table += "\\midrule \n"
                    table += r"\multirow{2}*{" + display_name + "}   & " + dim
                else:
                    table += " & " + dim
                for cell_index, value in enumerate(value_table[row_index]):
                    if value is None:
                        value = "& - "
                    elif "best" in value:
                        m, std = value[0], value[1]
                        if cell_index <= 3:
                            value = r"& $\mathbf{ " + f"{m:04.1f}" + r"_{\pm " + f"{std:04.1f}" + r"}}$"
                        else:
                            value = r"& $\mathbf{ " + f"{m:0.1f}" + r"_{\pm " + f"{std:0.1f}" + r"}}$"
                    else:
                        m, std = value[0], value[1]
                        pvalue = value["pvalue"]
                        if pvalue > 0.05:
                            pvalue = f"^*"
                        else:
                            pvalue = ""
                        if cell_index <= 3:
                            value = r"& $" + f"{m:04.1f}" + r"_{\pm " + f"{std:04.1f}" + r"}" + f"{pvalue}" + "$"
                        else:
                            value = r"& $" + f"{m:0.1f}" + r"_{\pm " + f"{std:0.1f}" + r"}" + f"{pvalue}" + "$"
                    table += value

                table += "\\\\ \n"
                row_index += 1

        table += r"""
                    \bottomrule
        \end{tabular}
        }
        \begin{tablenotes}
            \item Y/N: means whether to fine-tune with auxiliary labels
            \item \quad p : using pre-trained model
            \item \quad s : using pseudo-label
            \item w/ * : compariable in t-test;
            \item w/o*: significant improvements compare with best result
            $footnote$
        \end{tablenotes}
    \end{threeparttable}
\end{table}
""".replace(
            "$footnote$", footnote
        )
        # save to file
        os.makedirs(f"{MICCAI_LATEX_DOC_ROOT}", exist_ok=True)
        with open(f"{MICCAI_LATEX_DOC_ROOT}/{table_id}-result-{metric_index}.tex", "w", encoding="utf-8") as f:
            f.write(table)


def miccai_latex_table_avage2(
    all_result: dict[str, ExperimentResult],
    table_id="overview",
    show_items=LATEX_DISPLAY_ORDER,
    title="基准实验结果",
    title_en="Model experiment results",
    footnote="",
    split_keys=set("txtsam"),
) -> str:
    show_items = [item for item in show_items if item[0] in all_result]
    for metric_index, metric_list in enumerate([["dice", "iou", "msd", "hd"]]):
        table = r"""
    \begin{table}[!htpb]
        \caption{$title_en$}
        \label{tab:$table_id$-result-$index$}
        \centering
        \begin{threeparttable}[b]
            \scalebox{0.63}{
            \begin{tabular}{ccccccc}
                \toprule
                \multicolumn{3}{c}{Experiments} & Dice & Iou & Msd & HD  \\
        """
        table = (
            table.replace("$index$", str(metric_index))
            .replace("$result$", "\&".join(metric_list))
            .replace("$metric1$", metric_list[0])
            .replace("$metric2$", metric_list[1])
            .replace("$metric3$", metric_list[2])
            .replace("$metric4$", metric_list[3])
            .replace("$table_id$", table_id)
            .replace("$title$", title)
            .replace("$title_en$", title_en)
        )

        # 计算出每个格子的数据
        # experiment, 2d, N, [metric1, metric2,...]
        # experiment, 2d, Y, [metric1, metric2,...]
        # experiment, 3d, N, [metric1, metric2,...]
        # experiment, 3d, Y, [metric1, metric2,...]
        value_table = []
        for name, _display_name in show_items:
            result = all_result[name]
            for experiment_key in ["2d", "2daux", "3d", "3daux"]:
                row = []
                for metric in metric_list:
                    lower_better = metric == "msd" or metric == "hd"
                    value_list = []
                    for setting_id in range(5):
                        experiments: dict[str, dict[str, torch.Tensor]] = result.settings[setting_id]
                        experiment = experiments[experiment_key]
                        if metric in experiment:
                            experiment = experiment[metric]
                            value_list.append(experiment)

                    if len(value_list) == 0:
                        cell = None
                    else:
                        value_list = torch.cat(value_list)
                        value_list = value_list[torch.isfinite(value_list)]
                        if value_list.numel() == 0:
                            cell = None
                        else:
                            cell = {
                                0: value_list.mean().item(),
                                1: value_list.std().item(),
                                "lower_better": lower_better,
                            }
                    row.append(cell)
                value_table.append(row)

        for cell_index in range(len(metric_list)):
            best_value, best_item = None, None
            for row in value_table:
                current = row[cell_index]
                if current is None:
                    continue
                if best_item is None:
                    best_value = current[0]
                    best_item = current
                elif current["lower_better"] and current[0] < best_value:
                    best_value = current[0]
                    best_item = current
                elif (not current["lower_better"]) and current[0] > best_value:
                    best_value = current[0]
                    best_item = current
            if best_item is not None:
                best_item["best"] = True

        # print(value_table)
        row_index = 0
        for name, display_name in show_items:
            for dim in ["2D", "3D"]:
                for aux in ["N", "Y"]:
                    if row_index % 4 == 0:
                        table += "\\midrule \n"
                        if name in split_keys:
                            table += "\\midrule \n"
                        table += r"\multirow{4}*{" + display_name + "}"

                    if row_index % 2 == 0:
                        if dim == "3D":
                            table += "\cmidrule(lr){2-7} "
                        table += r" & \multirow{2}*{" + dim + "}  "
                    else:
                        table += " & "
                    table += " & " + aux
                    for value in value_table[row_index]:
                        if value is None:
                            value = "& - "
                        elif "best" in value:
                            m, std = value[0], value[1]
                            value = r"& $\mathbf{ " + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}}$"
                        else:
                            m, std = value[0], value[1]
                            value = r"& $" + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}$"
                        table += value

                    table += "\\\\ \n"
                    row_index += 1

        table += r"""
                    \bottomrule
        \end{tabular}
        }
        \begin{tablenotes}
            \item [Y/N] means whether to fine-tune with auxiliary labels
            \item [p] means using pre-trained model
            $footnote$
        \end{tablenotes}
    \end{threeparttable}
\end{table}
""".replace(
            "$footnote$", footnote
        )
        # save to file
        with open(f"{MICCAI_LATEX_DOC_ROOT}/{table_id}-result-{metric_index}.tex", "w", encoding="utf-8") as f:
            f.write(table)


def plot_table_avage_zeroshot(all_result: dict[str, ExperimentResult]) -> str:
    table = f"""<table class="pure-table metric-table">
    <caption>zeroshot avage metric</caption><tbody>
    <tr>
    <th rowspan=3 class="bb center">&nbsp;</th>
    """
    metric_list = ["dice", "iou", "msd", "hd"]
    for metric in metric_list:
        table += f"<th colspan=4 class='bb center'>{metric}</th>"
    table += "</tr><tr>"
    for _ in range(len(metric_list)):
        table += f"<td colspan=2 class='bb center'>主数据</td><td colspan=2 class='bb center'>主数据+辅助数据</td>"
    table += "</tr><tr>"
    for _ in range(len(metric_list)):
        table += f"<td class='bb center'>2D</td><td class='bb center'>3D</td>"
    table += "</tr>"

    value_table = {}
    for name, result in all_result.items():
        if name not in value_table:
            value_table[name] = []
        for metric in metric_list:
            lower_better = metric == "msd" or metric == "hd"
            for experiment_key in ["2d", "3d", "2daux", "3daux"]:
                value_list = []
                for setting_id in range(5):
                    experiments: dict[str, dict[str, torch.Tensor]] = result.zeroshot_settings[setting_id]
                    experiment = experiments[experiment_key]
                    if metric in experiment:
                        experiment = experiment[metric]
                        # 只用非nan/inf的数据
                        experiment = experiment[~torch.isnan(experiment)]
                        experiment = experiment[~torch.isinf(experiment)]
                        value_list.append(experiment)
                if len(value_list) == 0:
                    value_table[name].append(None)
                else:
                    value_list = torch.cat(value_list)
                    item = {
                        0: value_list.mean().item(),
                        1: value_list.std().item(),
                        "lower_better": lower_better,
                    }
                    value_table[name].append(item)
    for i in range(2 * len(metric_list)):
        best_value, best_item = None, None
        for _, value in value_table.items():
            current = value[i]
            if current is None:
                continue
            if best_item is None:
                best_value = current[0]
                best_item = current
            elif current["lower_better"] and current[0] < best_value:
                best_value = current[0]
                best_item = current
            elif (not current["lower_better"]) and current[0] > best_value:
                best_value = current[0]
                best_item = current
        if best_item is not None:
            best_item["best"] = True

    # print(value_table)
    for name in all_result.keys():
        table += f"<tr class='data'><td>{name}</td>"
        for idx, value in enumerate(value_table[name]):
            if value is None:
                value = "<center>-<center>"
            elif "best" in value:
                m, std = value[0], value[1]
                value = f"<pre class=best>{m:>7.4f}<sub>±{std:>6.4f}</sub></pre>"
            else:
                m, std = value[0], value[1]
                value = f"<pre>{m:>7.4f}<sub>±{std:>6.4f}</sub></pre>"

            cls = "d2d" if idx % 2 == 0 else "d3d"
            table += f"<td class='{cls}'>{value}</td>"

        table += "</tr>"

    table += "</tbody></table>"

    return table


def latex_table_avage_zeroshot(all_result: dict[str, ExperimentResult]) -> str:
    show_items = [item for item in all_result if item[0] in LATEX_DISPLAY_ORDER]
    metric_list = ["dice", "iou", "msd", "hd"]
    table = r"""
    \begin{table}[!htpb]
        \bicaption{zeroshot实验结果}{zeroshot experiment results}
        \label{tab:zeroshot-result}
        \centering
        \begin{threeparttable}[b]
            \begin{tabular}{cccccc}
                \toprule
                \multicolumn{2}{c}{zeroshot实验} & $metrics$ \\
        """
    table = table.replace("$metrics$", "&".join(metric_list))

    # 计算出每个格子的数据
    # experiment, 2d, [metric1, metric2,...]
    # experiment, 3d, [metric1, metric2,...]
    value_table = []
    for name, _display_name in show_items:
        result = all_result[name]
        row = []
        for metric in metric_list:
            value_statistic = {}
            lower_better = metric == "msd" or metric == "hd"
            for experiment_key in ["2daux"]:
                value_list = []
                for setting_id in range(5):
                    experiments: dict[str, dict[str, torch.Tensor]] = result.zeroshot_settings[setting_id]
                    experiment = experiments[experiment_key]
                    if metric in experiment:
                        experiment = experiment[metric]
                        value_list.append(experiment)
                if len(value_list) == 0:
                    value_statistic[experiment_key] = None
                else:
                    value_list = torch.cat(value_list)
                    item = {
                        0: value_list.mean().item(),
                        1: value_list.std().item(),
                        "lower_better": lower_better,
                    }
                    value_statistic[experiment_key] = item
            row.append(value_statistic["2daux"])
        value_table.append(row)
        row = []
        for metric in metric_list:
            value_statistic = {}
            lower_better = metric == "msd" or metric == "hd"
            for experiment_key in ["3daux"]:
                value_list = []
                for setting_id in range(5):
                    experiments: dict[str, dict[str, torch.Tensor]] = result.zeroshot_settings[setting_id]
                    experiment = experiments[experiment_key]
                    if metric in experiment:
                        experiment = experiment[metric]
                        value_list.append(experiment)
                if len(value_list) == 0:
                    value_statistic[experiment_key] = None
                else:
                    value_list = torch.cat(value_list)
                    item = {
                        0: value_list.mean().item(),
                        1: value_list.std().item(),
                        "lower_better": lower_better,
                    }
                    value_statistic[experiment_key] = item
            row.append(value_statistic["3daux"])
        value_table.append(row)

    for cell_index in range(1 * len(metric_list)):
        best_value, best_item = None, None
        for row in value_table:
            current = row[cell_index]
            if current is None:
                continue
            if best_item is None:
                best_value = current[0]
                best_item = current
            elif current["lower_better"] and current[0] < best_value:
                best_value = current[0]
                best_item = current
            elif (not current["lower_better"]) and current[0] > best_value:
                best_value = current[0]
                best_item = current
        if best_item is not None:
            best_item["best"] = True

    # print(value_table)
    row_index = 0
    for name, display_name in show_items:
        for dim in ["2D", "3D"]:
            if row_index % 2 == 0:
                table += "\\midrule \n"
                if name == "txtsam":
                    table += "\\midrule \n"
                table += r"\multirow{2}*{" + display_name + "}   & " + dim
            else:
                table += " & " + dim
            for value in value_table[row_index]:
                if value is None:
                    value = "& - "
                elif "best" in value:
                    m, std = value[0], value[1]
                    value = r"& $\mathbf{ " + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}}$"
                else:
                    m, std = value[0], value[1]
                    value = r"& $" + f"{m:0.4f}" + r"_{\pm " + f"{std:0.4f}" + r"}$"
                table += value

            table += "\\\\ \n"
            row_index += 1

    table += r"""
                    \bottomrule
        \end{tabular}
        \begin{tablenotes}
            \item [p] 代表使用预训练模型
            \item effsams和effsamti 使用\defEfficientSAM 的 ti和s预训练模型进行微调模型
        \end{tablenotes}
    \end{threeparttable}
\end{table}
"""
    # save to file
    os.makedirs(f"{LATEX_DOC_ROOT}/figures/experiments", exist_ok=True)
    with open(f"{LATEX_DOC_ROOT}/figures/experiments/zeroshot-result.tex", "w", encoding="utf-8") as f:
        f.write(table)


def plot_table_numbers(key, all_result: dict[str, ExperimentResult]) -> str:
    table_col_counts = [1 for _ in range(5)]
    for setting_id in range(5):
        count = 1
        for _, result in all_result.items():
            for experiment_key in ["2d", "3d", "2daux", "3daux"]:
                experiment = result.settings[setting_id][experiment_key]
                if key not in experiment:
                    continue
                experiment = experiment[key]
                if experiment.shape[0] > count:
                    count = experiment.shape[0]
        table_col_counts[setting_id] = count

    # 生成内层数据表
    table = f"""<table class="pure-table number-table">
    <tbody>
    <tr>
    """
    for setting_id in range(5):
        table += f"<td class='bb' colspan={table_col_counts[setting_id]}>fold-{setting_id}</col>"
    table += "</tr>"

    column_number_list = [[] for _ in range(sum(table_col_counts))]

    for name, result in all_result.items():
        for experiment_key in ["2d", "3d", "2daux", "3daux"]:
            table += f"<tr class='data'>"
            colume_index = 0
            for setting_id in range(5):
                experiment = result.settings[setting_id][experiment_key]
                if key in experiment:
                    experiment = experiment[key].tolist()
                else:
                    experiment = []

                for i in range(table_col_counts[setting_id]):
                    if i >= len(experiment):
                        value = "-"
                    else:
                        value = f"{experiment[i]:>7.4f}"
                        column_number_list[colume_index].append(experiment[i])

                    table += f"<td class='bb'>{value}</td>"
                    colume_index += 1
            table += f"</tr>"
    table += "<tr>"
    for colume_data in column_number_list:
        if len(colume_data) == 0:
            value = "-"
        else:
            value = mean(colume_data)
            value = f"{value:>7.4f}"
        table += f"<td class='bb'>{value}</td>"

    table += "</tr>"
    table += "</tbody></table>"

    # 生成外层总表
    content_line_count = 1 + len(all_result.keys()) * 4 + 1
    table = f"""<table class="pure-table number-table">
    <caption>all numbers: {key}</caption><tbody>
    <tr>
        <td colspan=3 class="bb center">{key}</td>
        <td rowspan={content_line_count} class='bb center number-content'><div>{table}<div></td>
    </tr>
    """
    for key in all_result.keys():
        table += f"""<tr>
            <td rowspan=4 class='bb'>{key}</td>
            <td rowspan=2 class='bb'>主数据</td>
            <td class='bb'>2D</td>
        </tr>
        <tr> <td class='bb'>3D</td> </tr>
        <tr> 
            <td rowspan=2 class='bb'>主数据+辅助数据</td>
            <td class='bb'>2D</td>
        </tr>
        <tr> <td class='bb'>3D</td> </tr>
        """

        table += "</tr>"
    #
    table += """
    <tr><td colspan=3 class='bb center'>平均</td></tr>
    </tbody></table>"""

    return table


def plot_table_ttest(metric, all_result: dict[str, ExperimentResult]) -> str:
    # 生成内层数据表
    table = f"""<table class="pure-table number-table">
    <tbody>
    """
    table += "<tr>"
    for name, result in all_result.items():
        table += f"""
            <td colspan=4 class='bb center'>{name}</td>
        """
    table += "</tr>"
    table += "<tr>"
    for name, result in all_result.items():
        table += f"""
            <td colspan=2 class='bb center'>主数据</td>
            <td colspan=2 class='bb center'>主数据+辅助数据</td>
        """
    table += "</tr>"
    table += "<tr>"
    for name, result in all_result.items():
        table += f"""
            <td class='bb center'>2D</td>
            <td class='bb center'>3D</td>
            <td class='bb center'>2D</td>
            <td class='bb center'>3D</td>
        """
    table += "</tr>"

    ## 收集数据
    table_col_counts = [1 for _ in range(5)]
    for setting_id in range(5):
        count = 1
        for _, result in all_result.items():
            for experiment_key in ["2d", "3d", "2daux", "3daux"]:
                experiment = result.settings[setting_id][experiment_key]
                if metric not in experiment:
                    continue
                experiment = experiment[metric]
                if experiment.shape[0] > count:
                    count = experiment.shape[0]
        table_col_counts[setting_id] = count

    column_number_list = [[] for _ in range(sum(table_col_counts))]

    for name, result in all_result.items():
        for experiment_key in ["2d", "3d", "2daux", "3daux"]:
            colume_index = 0
            for setting_id in range(5):
                experiment = result.settings[setting_id][experiment_key]
                if metric in experiment:
                    experiment = experiment[metric].tolist()
                else:
                    experiment = []

                for i in range(table_col_counts[setting_id]):
                    if i >= len(experiment):
                        column_number_list[colume_index].append(None)
                    else:
                        column_number_list[colume_index].append(experiment[i])

                    colume_index += 1

    lower_better = metric == "msd" or metric == "hd"
    for src_index in range(len(all_result.items()) * 4):
        table += f"<tr class='data'>"
        for tgt_index in range(len(all_result.items()) * 4):
            if src_index == tgt_index:
                value = "<td class='bb center'>\\</td>"
            else:
                src, target = [], []
                for col in column_number_list:
                    if col[src_index] is not None and col[tgt_index] is not None:
                        src.append(col[src_index])
                        target.append(col[tgt_index])
                if len(src) == 0:
                    value = "<td class='bb center'>-</td>"
                else:
                    ttest_result = ttest_rel(target, src)
                    statistic, pvalue = ttest_result.statistic, ttest_result.pvalue
                    if pvalue > 0.05:
                        style = "gray"
                    elif lower_better and statistic <= 0:
                        style = "positive"
                    elif not lower_better and statistic >= 0:
                        style = "positive"
                    else:
                        style = ""

                    value = f"<td class='bb'><pre class='{style}'>{statistic:>8.4f}<sub>{pvalue:>6.4f}</sub></pre></td>"

            table += value
        table += f"</tr>"

    table += "</tbody></table>"

    # 生成外层总表
    content_line_count = 1 + len(all_result.keys()) * 4
    table = f"""<table class="pure-table number-table">
    <caption>t-test: {metric}</caption><tbody>
    <tr>
        <td colspan=3 rwospan=3 class="bb center" height='112px'>t-test: {metric}</td>
        <td rowspan={content_line_count} class='bb center number-content'><div>{table}<div></td>
    </tr>
    """
    for metric in all_result.keys():
        table += f"""<tr>
            <td rowspan=4 class='bb'>{metric}</td>
            <td rowspan=2 class='bb'>主数据</td>
            <td class='bb'>2D</td>
        </tr>
        <tr> <td class='bb'>3D</td> </tr>
        <tr> 
            <td rowspan=2 class='bb'>主数据+辅助数据</td>
            <td class='bb'>2D</td>
        </tr>
        <tr> <td class='bb'>3D</td> </tr>
        """

        table += "</tr>"
    #
    table += """
    </tbody></table>"""

    return table


def latex_table_ttest(
    metric,
    current_experiment_index,
    all_result: dict[str, ExperimentResult],
    table_id="result",
    show_items=LATEX_DISPLAY_ORDER,
    footnote="",
    scale=1.0,
):
    current_name, display_name = show_items[current_experiment_index]
    show_items = [item for item in show_items if item[0] in all_result]
    if current_name not in all_result:
        return
    current_experiment_index = show_items.index((current_name, display_name))

    # 生成内层数据表
    table = r"""
    \begin{table}[!htpb]
        \caption{$display_name$和其他模型的t-test（$metrics$指标）}
        \label{tab:ttest-$table_id$-$metrics$-$current_name$}
        \centering
        \begin{threeparttable}[b]
            \scalebox{$scale$}{
            \begin{tabular}{ccccccc}
                \toprule
                \multicolumn{3}{c}{\multirow{2}{*}{$display_name$}} & \multicolumn{2}{c}{不使用辅助数据} & \multicolumn{2}{c}{使用辅助数据} \\
                \cmidrule(lr){4-5} \cmidrule(lr){6-7}
                \multicolumn{3}{c}{}                                & 2D        & 3D        & 2D     &   3d \\
        """
    table = (
        table.replace("$metrics$", metric)
        .replace("$display_name$", display_name)
        .replace("$current_name$", current_name)
        .replace("$table_id$", table_id)
        .replace("$scale$", f"{scale:0.2f}")
    )

    ## 收集数据
    table_col_counts = [1 for _ in range(5)]
    for setting_id in range(5):
        count = 1
        for _, result in all_result.items():
            for experiment_key in ["2d", "3d", "2daux", "3daux"]:
                experiment = result.settings[setting_id][experiment_key]
                if metric not in experiment:
                    continue
                experiment = experiment[metric]
                if experiment.shape[0] > count:
                    count = experiment.shape[0]
        table_col_counts[setting_id] = count

    column_number_list = [[] for _ in range(sum(table_col_counts))]

    for name, _diaplay_name in show_items:
        result = all_result[name]
        for experiment_key in ["2d", "3d", "2daux", "3daux"]:
            subject_index = 0
            for setting_id in range(5):
                experiment = result.settings[setting_id][experiment_key]
                if metric in experiment:
                    experiment = experiment[metric].tolist()
                else:
                    experiment = []

                for i in range(table_col_counts[setting_id]):
                    if i >= len(experiment):
                        column_number_list[subject_index].append(None)
                    else:
                        column_number_list[subject_index].append(experiment[i])

                    subject_index += 1

    lower_better = metric == "msd" or metric == "hd"
    for src_index in range(len(show_items) * 4):
        if src_index % 4 == 0:
            _, current_display_name = show_items[src_index // 4]
            table += "\\midrule\n"
            table += r"\multirow{4}*{" + current_display_name + r"}              & \multirow{2}*{N} & 2D" + "\n"
        elif src_index % 4 == 2:
            table += r"\cmidrule(lr){2-7}               & \multirow{2}*{Y} & 2D" + "\n"
        else:
            table += r"               &               & 3D" + "\n"
        for tgt_index in range(current_experiment_index * 4, current_experiment_index * 4 + 4):
            if src_index == tgt_index:
                value = r" & $\backslash$ "
            else:
                src, target = [], []
                for subject in column_number_list:
                    if subject[src_index] is not None and subject[tgt_index] is not None:
                        src.append(subject[src_index])
                        target.append(subject[tgt_index])
                if len(src) == 0:
                    value = "& - "
                else:
                    ttest_result = ttest_rel(target, src)
                    statistic, pvalue = ttest_result.statistic, ttest_result.pvalue
                    if pvalue > 0.05:
                        value = r"& $\color{grey}" + f"{statistic:0.4f}" + r"_{ " + f"{pvalue:0.4f}" + r"}$"
                    elif (lower_better and statistic <= 0) or (not lower_better and statistic >= 0):
                        value = r"& $\mathbf{ " + f"{statistic:0.4f}" + r"_{ " + f"{pvalue:0.4f}" + r"}}$"
                    else:
                        value = r"& $" + f"{statistic:0.4f}" + r"_{ " + f"{pvalue:0.4f}" + r"}$"
            table += value
        table += "\\\\\n"

    if footnote is not None and len(footnote) > 0:
        footnote = r""" \begin{tablenotes}
        $footnote$
        \end{tablenotes} """.replace(
            "$footnote$", footnote
        ).replace(
            "$scale$", f"{scale:0.2f}"
        )

    table += r"""            \bottomrule
        \end{tabular}
        }
        $footnote$
    \end{threeparttable}
\end{table}""".replace(
        "$footnote$", footnote
    )

    # save to file
    os.makedirs(f"{LATEX_DOC_ROOT}/figures/ttest", exist_ok=True)
    with open(
        f"{LATEX_DOC_ROOT}/figures/appendex/ttest-{table_id}-{metric}-{current_name}.tex", "w", encoding="utf-8"
    ) as f:
        f.write(table)


def run(args: List[str] | None):
    root_dir = ".checkpoint"

    setting_list = ["baseline-pa-0", "baseline-pa-1", "baseline-pa-2", "baseline-pa-3", "baseline-pa-4"]

    all_result: dict[str, ExperimentResult] = {}

    for setting_id, setting in enumerate(setting_list):
        experiment_list = glob("*", root_dir=f"{root_dir}/{setting}")
        for experiment_name in experiment_list:
            experiment_root = f"{root_dir}/{setting}/{experiment_name}"

            # 查找valid最优的epoch
            # print(experiment_root)
            best_epoch = find_best_epoch(experiment_root)
            if len(best_epoch) == 0:
                continue

            name_dict = ExperimentName(experiment_name)
            if len(name_dict.extra) > 0:  # 可能为预训练的临时数据
                continue
            name = name_dict.name
            if name not in all_result:
                all_result[name] = ExperimentResult()
            current_result = all_result[name]
            current_result = current_result.get_result(setting_id, name_dict)

            current_result.update(torch.load(best_epoch))

            zeroshot_epoch = best_epoch.replace("/test-metrics.pt", "/zeroshot-metrics.pt")
            if os.path.exists(zeroshot_epoch):
                current_result = all_result[name]
                current_result = current_result.get_zeroshot_result(setting_id, name_dict)
                current_result.update(torch.load(zeroshot_epoch))

    mark_best(all_result)
    all_result = OrderedDict(sorted(all_result.items()))

    # save all_result to json
    with open("result.json", "w") as f:
        f.write(json.dumps([(k, f"{v}") for k, v in all_result.items()], indent=4))

    # print(all_result["txtsam[ps]"])
    # print(all_result)
    # 绘制表格
    document = """<html>

<style>
    html {
        font-family: sans-serif;
        -ms-text-size-adjust: 100%;
        -webkit-text-size-adjust: 100%;
        background-color: #fff;
    }

    body { margin: 10px; }

    table { border-collapse: collapse; border-spacing: 0; }

    td, th { padding: 0; }


    .center {text-align:center;}

    .pure-table {
        border-collapse: collapse;
        border-spacing: 0;
        empty-cells: show;
        border: 1px solid #cbcbcb;
    }

    .pure-table caption {
        color: #000;
        font: italic 85%/1 arial, sans-serif;
        padding: 1em 0;
        text-align: left;
        font-size: 12px;
    }

    .pure-table td,
    .pure-table th {
        border-left: 1px solid #cbcbcb;
        border-width: 0 0 0 1px;
        font-size: 15px;
        margin: 0;
        overflow: visible;
        padding: .5em 1em;
    }
    .pure-table td sub{font-size:11px; padding-left:2px;}

    .pure-table .bb { border-bottom: solid 1px #cbcbcb;}

    .pure-table thead {
        background-color: #e0e0e0;
        color: #000;
        text-align: left;
        vertical-align: bottom;
    }
    .pure-table  pre.best { font-weight: 900; background-color: #eee; }
    .pure-table .data:hover { background-color: #ffeb3b4f; }
    .pure-table .data:hover.d2d .d2d, .pure-table .data:hover.d3d .d3d { color: red; }

    .number-table {width: calc(100vw - 80px);}
    .number-table .number-content {width: calc(100vw - 250px - 80px); padding: 0;}
    .number-table .number-content>div {width: calc(100vw - 250px - 80px); overflow:scroll;}
    .gray {color: #bbb;}
    .positive { font-weight: 900;}
</style>
    <body>"""
    document += f"""<h2>模型结果 <sub>生成时间：[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]</sub></h2>"""
    document += "<h2>以下为各指标的加权平均成绩</h2>"
    document += plot_table_avage(all_result)
    document += "<h2>以下为zeroshot各指标的加权平均成绩</h2>"
    document += plot_table_avage_zeroshot(all_result)
    document += "<h2>以下为各指标的每个cross validation成绩</h2>"
    for key in ["dice", "iou", "msd", "hd"]:
        document += plot_table_detail(key, all_result)
    document += "<h2>以下为zeroshot各指标的每个cross validation成绩</h2>"
    for key in ["dice", "iou", "msd", "hd"]:
        document += plot_table_detail_zeroshot(key, all_result)
    document += "<h2>以下为每个case在每个模型下面的测试成绩(每一列代表一个case)</h2>"
    for key in ["dice", "iou", "msd", "hd"]:
        document += plot_table_numbers(key, all_result)
    document += "<h2>各模型的t-test矩阵(列-行, 加黑的数字表示:'列' 比 '行'优)</h2>"
    for key in ["dice", "iou", "msd", "hd"]:
        document += plot_table_ttest(key, all_result)
    document += """<br/></body>
    <script>
    function onmove(e){
        target = e.target
        cls_name = null
        while(target && target.tagName != 'TR') { 
            if(target.tagName == 'TD'){cls_name = target.className}
            target = target.parentElement;
        }
        if(!cls_name || !target || target.classList.contains(cls_name)) {return}
        target.classList.remove('d2d')
        target.classList.remove('d3d')
        if(cls_name.includes('d2d')) {
            target.classList.add('d2d')
        }
        if(cls_name.includes('d3d')) {
            target.classList.add('d3d')
        }

    }
    for(table of document.getElementsByClassName('metric-table')) { table.addEventListener('mousemove',onmove); }
    </script>
    </html>"""

    with open("result.html", "w") as f:
        f.write(document)

    # 生成latex
    miccai_latex_table_avage1(
        all_result,
        "baseline",
        [ALL_EXPERIMENT_DISPLAY_ORDER[i] for i in [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12]],
        split_keys=["txtsam"],
    )
    miccai_latex_table_avage2(all_result, "baseline-all", LATEX_DISPLAY_ORDER)

    latex_table_avage(all_result, "overview", LATEX_DISPLAY_ORDER)
    latex_table_avage(
        all_result,
        "pseudo",
        [ALL_EXPERIMENT_DISPLAY_ORDER[i] for i in [4, 3, 9, 11, 10, 12]],
        "伪标签增强数据集实验结果",
        "pseudo result",
        footnote=""" \item [s] 代表使用伪标签数据集
        \item 通过两两对比，判断增加伪标签数据集对模型性能是否有提升
        """,
        split_keys=set(["txtsam", "txtsam[p]"]),
    )
    latex_table_avage_zeroshot(all_result)
    table_detail_part1 = [ALL_EXPERIMENT_DISPLAY_ORDER[i] for i in [0, 1, 2, 5, 7, 8, 12]]
    table_detail_part2 = [ALL_EXPERIMENT_DISPLAY_ORDER[i] for i in [6, 4, 3, 9, 11, 10, 12]]
    for key in ["dice", "iou", "msd", "hd"]:
        latex_table_detail(key, all_result, table_detail_part1, "1")
        latex_table_detail(key, all_result, table_detail_part2, "2")
        latex_table_detail_zeroshot(key, all_result)
        for idx in range(len(LATEX_DISPLAY_ORDER)):
            latex_table_ttest(key, idx, all_result, scale=0.78)
    experiment_ttest = [ALL_EXPERIMENT_DISPLAY_ORDER[i] for i in [0, 1, 2, 4, 5, 6, 9, 10]]
    latex_table_ttest(
        "dice",
        7,
        all_result,
        "experiment-base",
        experiment_ttest,
        footnote="""
        \item \scalebox{$scale$}{1. 表格展示了t检验的T值(正常显示)和对应的p-value(下标)，并精确到小数点后4位。}
        \item \scalebox{$scale$}{2. 灰色表示p值大于0.05，无法拒绝原假设，即结果不显著；加黑部分代表当前模型结果较优。}
        \item \scalebox{$scale$}{3. 非加粗字体表示当前模型(列)在该指标上劣于对比模型（行）。}
""",
        scale=0.87,
    )
    experiment_ttest = [ALL_EXPERIMENT_DISPLAY_ORDER[i] for i in [4, 3, 9, 11, 10, 12]]
    latex_table_ttest(
        "dice",
        5,
        all_result,
        "experiment-pseudo",
        experiment_ttest,
        footnote="""
        \item 1. 表格展示了t检验的T值(正常显示)和对应的p-value(下标)，并精确到小数点后4位。
        \item 2. 灰色表示p值大于0.05，无法拒绝原假设，即结果不显著；加黑部分代表当前模型结果较优。
        \item 3. 非加粗字体表示当前模型(列)在该指标上劣于对比模型（行）。
""",
    )
