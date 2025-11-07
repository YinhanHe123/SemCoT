import argparse
from datetime import datetime
import gc
import os
import torch
from models.coconut import Coconut
from models.codi import CODI
from models.icot_si import ICoT_SI
from models.pause import Pause
from models.semcot import ContempGen, CustomST
from data.cot_datasets import load_datasets
from models.softcot import SoftCoT
from training.train_coconut import run_coconut_inference, train_coconut_model
from training.train_codi import run_codi_inference, train_codi_model
from training.train_icot_si import run_icot_si_inference, train_icot_si_model
from training.train_pause import run_pause_inference, train_pause_model
from training.train_semcot import (
    load_reasoning_hidden,
    run_semcot_inference,
    train_contemp_gen,
    train_custom_st,
    update_reasoning_hidden,
)
from training.train_softcot import run_softcot_inference, train_softcot_model
from utils.logging import Logger
import utils.utils as utils


def run_baseline(logger, args, train_data, eval_data):
    if args.baseline == "softcot":
        model_path = train_softcot_model(logger, args, train_data, eval_data)
        model = SoftCoT.from_pretrained(model_path).to(args.device)
        model.eval()
        res = run_softcot_inference(logger, model, eval_data, args)
    elif args.baseline == "codi":
        model_path = train_codi_model(logger, args, train_data, eval_data)
        model = CODI.from_pretrained(model_path).to(args.device)
        model.eval()
        res = run_codi_inference(logger, model, eval_data, args)
    elif args.baseline == "icot_si":
        model_path = train_icot_si_model(logger, args, train_data, eval_data)
        model = ICoT_SI.from_pretrained(model_path).to(args.device)
        model.eval()
        res = run_icot_si_inference(logger, model, eval_data, args)
    elif args.baseline == "pause":
        model_path = train_pause_model(logger, args, train_data, eval_data)
        model = Pause.from_pretrained(model_path).to(args.device)
        model.eval()
        res = run_pause_inference(logger, model, eval_data, args)
    elif args.baseline == "coconut":
        model_path = train_coconut_model(logger, args, train_data, eval_data)
        model = Coconut.from_pretrained(model_path).to(args.device)
        model.eval()
        res = run_coconut_inference(logger, model, eval_data, args)
    model = model.cpu()
    del model
    return res


def load_model_configs(args):
    if args.config == "small":
        args.teacher_model_name = "meta-llama/Llama-2-7b-chat-hf"
        args.student_model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
        args.teacher_hid_dim = 4096
    elif args.config == "mistral":
        args.teacher_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        args.student_model_name = "optimum/mistral-1.1b-testing"
        args.teacher_hid_dim = 4096
    elif args.config == "qwen":
        args.teacher_model_name = "Qwen/Qwen2.5-7B-Instruct"
        args.student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        args.teacher_hid_dim = 3584


def load_file_paths(args):
    model_type = args.variation if args.mode == "semcot" else args.baseline
    args.base_path = os.path.join(
        ".",
        "results",
        args.mode,
        model_type,
        args.config,
        args.dataset,
    )
    args.logging_path = os.path.join(args.base_path, "logs")
    args.result_path = os.path.join(args.base_path, "results")
    args.experiment_name = (
        f"{args.mode}_{model_type}_{args.seed}_{args.dataset}_{args.config}"
    )
    os.makedirs(args.logging_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="SemCoT Main Experiments")
    parser.add_argument("--num_exps", type=int, default=3, help="Number of experiments")
    parser.add_argument(
        "--use_best_params", action="store_true", help="Use best parameters"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["semcot", "baseline", "generate_data"],
        default="semcot",
        help="Operation mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["small", "mistral", "qwen"],
        default="small",
        help="Configuration name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "svamp", "multiarith", "commonsense_qa", "coin_flip"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["pause", "icot_si", "codi", "softcot", "coconut"],
        help="baseline type",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--variation",
        type=str,
        default="vanilla",
        choices=[
            "vanilla",
            "no_sentence_transformer",
            "no_l_reason",
            "no_warmup",
            "no_small_contemp_gen",
            "param_analysis"
        ],
        help="Variation of the semcot model to use",
    )

    # Hyperparamters
    parser.add_argument(
        "--train_max_contemp_tokens",
        type=int,
        default=5,
        help="max number of contemp tokens for training semcot",
    )
    parser.add_argument(
        "--eval_max_contemp_tokens",
        type=int,
        default=1,
        help="max number of contemp tokens for evaluating semcot",
    )
    parser.add_argument(
        "--eval_temp", type=float, default=0.7, help="Temperature for evaluation"
    )
    parser.add_argument(
        "--max_seq_len",
        type=float,
        default=512,
        help="Max sequence length for tokenization",
    )
    parser.add_argument("--batch_size", type=float, default=4, help="Batch size")
    parser.add_argument(
        "--st_linear_lr",
        "-stllr",
        type=float,
        default=1e-4,
        help="Linear layer learning rate for sentence transformer",
    )
    parser.add_argument(
        "--st_linear_wd",
        "-stlwd",
        type=float,
        default=1e-3,
        help="Linear layer weight decay for sentence transformer",
    )
    parser.add_argument(
        "--st_linear_epochs",
        "-stle",
        type=int,
        default=7,
        help="Linear layer number of epochs for sentence transformer",
    )
    parser.add_argument(
        "--st_llm_lr",
        "-stllmlr",
        type=float,
        default=1e-7,
        help="LLM learning rate for sentence transformer",
    )
    parser.add_argument(
        "--st_llm_wd",
        "-stllmwd",
        type=float,
        default=1e-5,
        help="LLM weight decay for sentence transformer",
    )
    parser.add_argument(
        "--st_llm_epochs",
        "-stllme",
        type=int,
        default=3,
        help="LLM number of epochs for sentence transformer",
    )
    parser.add_argument(
        "--cg_linear_lr",
        "-cgllr",
        type=float,
        default=1e-4,
        help="Linear layer learning rate for contemp generator",
    )
    parser.add_argument(
        "--cg_linear_wd",
        "-cglwd",
        type=float,
        default=1e-3,
        help="Linear layer weight decay for contemp generator",
    )
    parser.add_argument(
        "--cg_linear_epochs",
        "-cgle",
        type=int,
        default=7,
        help="Linear layer number of epochs for contemp generator",
    )
    parser.add_argument(
        "--cg_llm_lr",
        "-cgllmlr",
        type=float,
        default=1e-7,
        help="LLM learning rate  for contemp generator",
    )
    parser.add_argument(
        "--cg_llm_wd",
        "-cgllmwd",
        type=float,
        default=1e-5,
        help="LLM weight decay  for contemp generator",
    )
    parser.add_argument(
        "--cg_llm_epochs",
        "-cgllme",
        type=int,
        default=3,
        help="LLM number of epochs for contemp generator",
    )
    parser.add_argument(
        "--alpha", type=float, default=0, help="alpha hyperparameter"
    )

    # LoRA specific arguments
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="Rank for LoRA adapter"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=32, help="Alpha scaling factor for LoRA"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--contemp_gen_name",
        type=str,
        default=None,
        help="ContempGen pretrained model name",
    )
    return parser.parse_args()


def main(args):
    train_path = os.path.join(".", "datasets", args.dataset, f"train_{args.seed}.json")
    eval_path = os.path.join(".", "datasets", args.dataset, f"eval_{args.seed}.json")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_data, eval_data = load_datasets(args.dataset, train_path, eval_path)
    if args.mode == "generate_data":
        exit(0)
    if args.use_best_params:
        args.st_linear_lr = utils.BEST_PARAMS[args.config][args.dataset]["stllr"]
        args.st_linear_wd = utils.BEST_PARAMS[args.config][args.dataset]["stlwd"]
        args.st_llm_lr = utils.BEST_PARAMS[args.config][args.dataset]["stllmlr"]
        args.st_llm_wd = utils.BEST_PARAMS[args.config][args.dataset]["stllmwd"]
        args.st_linear_epochs = utils.BEST_PARAMS[args.config][args.dataset]["stle"]
        args.st_llm_epochs = utils.BEST_PARAMS[args.config][args.dataset]["stllme"]
        args.cg_linear_lr = utils.BEST_PARAMS[args.config][args.dataset]["cgllr"]
        args.cg_linear_wd = utils.BEST_PARAMS[args.config][args.dataset]["cglwd"]
        args.cg_llm_lr = utils.BEST_PARAMS[args.config][args.dataset]["cgllmlr"]
        args.cg_llm_wd = utils.BEST_PARAMS[args.config][args.dataset]["cgllmwd"]
        args.cg_linear_epochs = utils.BEST_PARAMS[args.config][args.dataset]["cgle"]
        args.cg_llm_epochs = utils.BEST_PARAMS[args.config][args.dataset]["cgllme"]
        if args.alpha == 0:
            args.alpha = utils.BEST_PARAMS[args.config][args.dataset]["alpha"]

    load_model_configs(args)
    load_file_paths(args)

    logger = Logger(
        log_dir=args.logging_path,
        experiment_name=f"{args.experiment_name}_{datetime.now()}",
    )
    logger.log_hyperparams(args.__dict__)

    if args.contemp_gen_name is None:
        for i in range(args.num_exps):
            args.model_save_path = os.path.join(args.base_path, f"saved_model_exp={i}")
            os.makedirs(args.model_save_path, exist_ok=True)
            if args.mode == "semcot":
                load_reasoning_hidden(
                    train_data,
                    args.teacher_model_name,
                    args.device,
                    args.max_seq_len,
                )
                load_reasoning_hidden(
                    eval_data,
                    args.teacher_model_name,
                    args.device,
                    args.max_seq_len,
                )

                custom_st = None
                if args.variation != "no_sentence_transformer":
                    st_model_path = train_custom_st(train_data, args, logger)
                    custom_st = CustomST.from_pretrained(st_model_path).to(args.device)
                    custom_st.eval()
                    for param in custom_st.parameters():
                        param.requires_grad = False
                    del param
                    update_reasoning_hidden(train_data, args.device, custom_st)
                    update_reasoning_hidden(eval_data, args.device, custom_st)

                # Train the contemplation generator
                cg_model_path = train_contemp_gen(
                    custom_st, train_data, eval_data, logger, args
                )
                contemp_gen = ContempGen.from_pretrained(cg_model_path).to(args.device)
                metrics = run_semcot_inference(logger, contemp_gen, eval_data, args)
                # Evaluate results
                for metric in metrics:
                    metric["exp_num"] = i
                    utils.append_to_jsonl_file(f"{args.result_path}/eval_res.jsonl", metric)
                if custom_st is not None:
                    custom_st = custom_st.cpu()
                contemp_gen = contemp_gen.cpu()
                del contemp_gen, custom_st
                gc.collect()
                torch.cuda.empty_cache()
            elif args.mode == "baseline":
                # Run baseline method
                metrics = run_baseline(logger, args, train_data, eval_data)
                # Evaluate results
                for metric in metrics:
                    metric["exp_num"] = i
                    utils.append_to_jsonl_file(f"{args.result_path}/eval_res.jsonl", metric)
    else:
        contemp_gen = ContempGen(
            args.student_model_name,
            args.teacher_hid_dim,
            args.variation,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            contemp_gen_name=args.contemp_gen_name,
        ).to(args.device)
        metrics = run_semcot_inference(logger, contemp_gen, eval_data, args)
        for metric in metrics:
            utils.append_to_jsonl_file(f"{args.result_path}/eval_res.jsonl", metric)

if __name__ == "__main__":
    args = parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    # Set random seed
    utils.set_seed(args.seed)
    main(args)
