import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_mortis import MortisConfig, MortisForCausalLM
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
        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from,
                                                     trust_remote_code=True)

    get_model_params(model, model.config)

    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="Mortis模型推理与对话")
    # 模型加载相关
    parser.add_argument("--load_from",
                        "-lf",
                        default="model", 
                        type=str, 
                        help="模型加载路径(model=torch原生权重, 其他路径=transformers格式)")
    parser.add_argument("--save_dir", 
                        "-sd",
                        default="out",
                        type=str,
                        help="模型权重目录")
    parser.add_argument("--weight", 
                        "-w",
                        default="full_sft", 
                        type=str, 
                        help="权重名称前缀(pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    # 模型架构相关
    parser.add_argument("--hidden_size", 
                        "-hs",
                        default=768,
                        type=int,
                        help="隐藏层维度(Base-84M=768)")
    parser.add_argument("--num_hidden_layers", 
                        "-nh",
                        default=12, 
                        type=int,
                        help="隐藏层数量(Base-84M=12)")
    parser.add_argument("--use_moe", 
                        "-um",
                        default=False,
                        action="store_true",
                        help="是否使用MOE架构")
    parser.add_argument("--inference_rope_scaling", 
                        "-irs",
                        default=False,
                        action="store_true",
                        help="启用ROPE位置编码外推(16 倍外推[从 2048 扩展到 32768]，仅解决位置编码问题)")
    # 生成参数相关
    parser.add_argument("--max_new_tokens", 
                        "-mnt",
                        default=8192,
                        type=int,
                        help="最大生成长度(注意：并非模型实际长文本能力)")
    parser.add_argument("--temperature",
                        "-t",
                        default=0.85,
                        type=float,
                        help="生成温度，控制随机性(0-1, 越大越随机)")
    parser.add_argument("--top_p",
                        default=0.85,
                        type=float,
                        help="nucleus采样阈值(0-1)")
    # 对话相关
    parser.add_argument("--history",
                        "-ht",
                        default=0,
                        type=int,
                        help="携带历史对话轮数(须为偶数，0表示不携带历史)")
    parser.add_argument("--show_speed",
                        "-sp",
                        default=1,
                        type=int,
                        help="显示decode速度(tokens/s)")
    parser.add_argument("--device",
                        "-d",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        type=str,
                        help="运行设备")
    args = parser.parse_args()
    
    # 预设测试问题
    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用python写一个计算斐波那契数列的函数",
        "解释一下'光合作用'的基本过程",
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食"
    ]
    
    # 对话历史，格式: [{"role": "use/assistant", "content": "..."}]
    conversation = [] 
    model, tokenizer = init_model(args)
    input_mode = int(input("[0]自动测试, [1]手动输入\n\n"))
    streamer = TextStreamer(tokenizer,
                            skip_prompt=True,
                            skip_special_tokens=True)
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("input: "), "")

    for prompt in prompt_iter:
        setup_seed(2026)
        
        if input_mode == 0:
            print(f"input: {prompt}")
        
        conversation = conversation[-args.history:] if args.history else []
        conversation.append({"role": "user", "content": prompt})
        templates = {
                "conversation": conversation,
                "tokenize": False,
                "add_generation_prompt": True,
        }
        if args.weight == "reason":
            templates["enable_thinking"] = True
        if args.weight != "pretrain":
            inputs = tokenizer.apply_chat_template(**templates)
        else:
            inputs = tokenizer.bos_token + prompt
        inputs = tokenizer(inputs,
                           return_tensors="pt",
                           truncation=True).to(args.device)

        print("output: ", end="")
        st = time.time()

        generation_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=1.0,
        )

        response = tokenizer.decode(
                generation_ids[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True,
        )

        conversation.append({"role": "assistant", "content": response})
        
        gen_tokens = len(generation_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed:
            print(f"\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n")
        else:
            print("\n\n")

if __name__ == "__main__":
    main()














