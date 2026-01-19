import time
import argsparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model import MortisConfig, MortisForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings("ignore")

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if "model" in args.load_from:
        # 原生pytorh权重
        model = MortisForCausalLM(MortisConfig(hidden_size=args.hidden_size,
                                               num_hidden_layers=args.num_hidden_layers,
                                               use_moe=bool(args.use_moe),
                                               inference_rope_scaling=args.inference_rope_scaling))
        moe_suffix = "_moe" if args.use_moe else ""
        ckp = f"./{args.save}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from,
                                                     trust_remote_code=True)

    get_model_params(model, model.config)

    return model.eval().to(args.device), tokenizer

def main():
    parser = argsparse.ArgumentParser(description="Mortis模型推理与对话")
    # 模型加载相关
    parser.add_argument("--load_from", 
                        default="model", 
                        type=str, 
                        help="模型加载路径(model=torch原生权重, 其他路径=transformers格式)")
    parser.add_argument("--save_dir", 
                        default="out",
                        type=str,
                        help="模型权重目录")
    parser.add_argument("--weight", 
                        default="full_sft", 
                        type=str, 
                        help=权重名称前缀(pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo))
    # 模型架构相关
    parser.add_argument("--hidden_size", 
                        default=768,
                        type=int,
                        help="隐藏层维度(768=Base-80M)")
    
    



