from run.baseline.model_arguments import ModelArguments
import torch


def get_config_train_labels(model_args: ModelArguments, device) -> tuple[list[torch.Tensor], list[list[int]]]:
    target_label_id_list_cuda = [
        torch.tensor(list(range(1, model_args.label_count))).to(device),  # all label
        torch.tensor(model_args.get_dataset_train_labels()).to(device),  # CMRI
        torch.tensor(model_args.get_aux_dataset_labels()).to(device),  # AUX
        torch.tensor(model_args.get_zeroshot_dataset_labels()).to(device),  # ZERO_SHOT
        torch.tensor(model_args.get_pseudo_output_labels()).to(device),  # pseudo
    ]
    target_label_id_list = [v.tolist() for v in target_label_id_list_cuda]

    print("train_labels[all,main,aux,zeroshot,pseudo]:", target_label_id_list)

    return target_label_id_list_cuda, target_label_id_list


def get_config_eval_labels(model_args: ModelArguments, device) -> tuple[list[torch.Tensor], list[list[int]]]:
    target_label_id_list_cuda = [
        torch.tensor(list(range(1, model_args.label_count))).to(device),  # all label
        torch.tensor(model_args.get_dataset_eval_labels()).to(device),  # CMRI
        torch.tensor(model_args.get_aux_dataset_labels()).to(device),  # AUX
        torch.tensor(model_args.get_zeroshot_dataset_labels()).to(device),  # ZERO_SHOT
        torch.tensor(model_args.get_pseudo_output_labels()).to(device),  # pseudo
    ]
    target_label_id_list = [v.tolist() for v in target_label_id_list_cuda]

    print("eval_labels[all,main,aux,zeroshot,pseudo]:", target_label_id_list)
    return target_label_id_list_cuda, target_label_id_list
