#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA-Factory 本地执行脚本（local_run 版本）
=========================================

该版本放置于 local_run/ 目录下：
- 工作目录(project_root) 自动定位为仓库根目录
- 零参数启动时默认读取 configs/runner_config.yaml
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# 计算仓库根目录（local_run 的上一级）
THIS_FILE = Path(__file__).resolve()
LOCAL_RUN_DIR = THIS_FILE.parent
PROJECT_ROOT = LOCAL_RUN_DIR.parent

# 确保可以直接导入源码（避免通过命令行）
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 直接调用源码入口
from llamafactory.train.tuner import run_exp, export_model  # type: ignore
from llamafactory.eval.evaluator import Evaluator  # type: ignore
from llamafactory.api.app import run_api  # type: ignore
from llamafactory.chat.chat_model import ChatModel, run_chat  # type: ignore
from llamafactory.webui.interface import run_web_demo, run_web_ui  # type: ignore

# 设置日志（写入 local_run 目录）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOCAL_RUN_DIR / 'logs' / 'llamafactory_runner.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class LLaMAFactoryRunner:
    """
    LLaMA-Factory本地执行器（local_run 版）
    
    这个类提供了在local_run目录下运行LLaMA-Factory的完整功能：
    - 训练：LoRA微调、全参数微调等
    - 推理：CLI聊天、Web聊天、API服务
    - 导出：模型合并和导出
    - 评估：模型性能评估
    
    所有输出文件都统一保存在local_run目录下，便于管理
    """

    def __init__(self):
        """
        初始化本地执行器
        设置所有必要的路径和目录结构
        """
        self.project_root = PROJECT_ROOT  # 项目根目录
        self.examples_dir = self.project_root / "examples"  # 示例配置目录
        self.data_dir = self.project_root / "data"  # 数据集目录
        # 将所有本地生成的配置与输出统一到 local_run 目录
        self.output_dir = LOCAL_RUN_DIR  # 输出目录
        self.saves_dir = LOCAL_RUN_DIR / "saves"  # 模型保存目录
        # 运行配置放在 configs 目录
        self.default_runner_cfg = LOCAL_RUN_DIR / "configs" / "runner_config.yaml"  # 默认运行配置

        # 创建必要的目录
        self._create_directories()

    def _create_directories(self):
        directories = [
            self.output_dir, 
            self.saves_dir,
            LOCAL_RUN_DIR / "configs",
            LOCAL_RUN_DIR / "logs",
            LOCAL_RUN_DIR / "models",
            LOCAL_RUN_DIR / "hf_cache"
        ]
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logger.info(f"确保目录存在: {directory}")

    def _load_task_args(self, task_config: Optional[str], overrides: dict) -> dict:
        """加载任务配置（yaml/json）并与 overrides 合并，overrides 优先生效。"""
        base: dict = {}
        if task_config:
            try:
                p = Path(task_config)
                with open(p, "r", encoding="utf-8") as f:
                    if p.suffix.lower() in {".yaml", ".yml"}:
                        base = yaml.safe_load(f) or {}
                    elif p.suffix.lower() == ".json":
                        base = json.load(f) or {}
            except Exception as e:
                logger.error(f"读取任务配置失败: {e}")
        base.update(overrides or {})
        return base

    def train(self, config_path: str = None, **kwargs) -> int:
        if config_path is None:
            config_path = self._create_default_train_config(**kwargs)
            logger.info(f"使用默认训练配置: {config_path}")
        else:
            logger.info(f"开始训练，配置文件: {config_path}")
        # 直接调用源码：将 yaml + 覆盖参数 传入 run_exp
        args = self._load_task_args(config_path, kwargs)
        run_exp(args=args)
        return 0

    def run_from_file(self, runner_config_path: Optional[str] = None) -> int:
        cfg_path = Path(runner_config_path) if runner_config_path else self.default_runner_cfg
        logger.info(f"自动模式：尝试从运行配置文件启动 -> {cfg_path}")
        if not cfg_path.exists():
            logger.warning("未发现运行配置文件，将回退到内置默认策略。")
            return self.run_auto_fallback()
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                if cfg_path.suffix.lower() in {".yml", ".yaml"}:
                    rcfg = yaml.safe_load(f) or {}
                else:
                    rcfg = json.load(f)
        except Exception as e:
            logger.error(f"读取运行配置失败：{e}")
            return 1
        command = (rcfg.get("command") or "train").strip()
        task_config = rcfg.get("config")
        # 若提供了相对路径的配置文件，则相对于 runner 配置文件所在目录解析
        if task_config and not Path(task_config).is_absolute():
            task_config = str((cfg_path.parent / task_config).resolve())
        args = rcfg.get("args") or {}
        logger.info(f"运行配置 -> command: {command}, config: {task_config}, args: {args}")
        if command == "train":
            return self.train(task_config, **args)
        if command == "eval":
            if not task_config:
                logger.error("eval 需要提供 config 路径。")
                return 1
            return self.eval(task_config, **args)
        if command == "export":
            if task_config:
                args = self._load_task_args(task_config, {})
                export_model(args=args)
                return 0
            required_keys = {"model_path", "adapter_path", "export_dir"}
            if not required_keys.issubset(set(args.keys())):
                logger.error("export 缺少必要参数：model_path/adapter_path/export_dir（或提供 config）。")
                return 1
            # 直接调用源码
            export_args = {
                "model_name_or_path": args.pop("model_path"),
                "adapter_name_or_path": args.pop("adapter_path"),
                "export_dir": args.pop("export_dir"),
            }
            export_args.update(args)
            export_model(args=export_args)
            return 0
        if command == "chat":
            if task_config:
                chat_args = self._load_task_args(task_config, {})
                return self._run_chat_with_args(chat_args)
            if "model_path" not in args:
                logger.error("chat 需要提供 model_path（或提供 config）。")
                return 1
            return self._run_chat_with_args(args)
        if command == "webchat":
            if task_config:
                # 目前 webchat 仅提供 CLI 入口，这里直接调用 Web Demo
                run_web_demo()
                return 0
            if "model_path" not in args:
                logger.error("webchat 需要提供 model_path（或提供 config）。")
                return 1
            run_web_demo()
            return 0
        if command == "api":
            if task_config:
                api_args = self._load_task_args(task_config, {})
                # 设置 host/port 并直接 run_api
                if "host" in api_args:
                    os.environ["API_HOST"] = str(api_args.get("host"))
                if "port" in api_args:
                    os.environ["API_PORT"] = str(api_args.get("port"))
                run_api()
                return 0
            if "model_path" not in args:
                logger.error("api 需要提供 model_path（或提供 config）。")
                return 1
            if "host" in args:
                os.environ["API_HOST"] = str(args.get("host"))
            if "port" in args:
                os.environ["API_PORT"] = str(args.get("port"))
            run_api()
            return 0
        if command == "webui":
            run_web_ui()
            return 0
        logger.error(f"未知的 command: {command}")
        return 1

    def run_auto_fallback(self) -> int:
        # 优先尝试 configs 目录下的本地训练配置
        preferred_local = LOCAL_RUN_DIR / "configs" / "qwen3_lora_sft.yaml"
        if preferred_local.exists():
            logger.info(f"使用本地训练配置：{preferred_local}")
            return self.train(str(preferred_local))
        # 次选 examples 目录下的示例配置
        preferred_example = self.examples_dir / "train_lora" / "qwen3_lora_sft.yaml"
        if preferred_example.exists():
            logger.info(f"使用示例训练配置：{preferred_example}")
            return self.train(str(preferred_example))
        logger.info("未找到本地或示例训练配置，使用内置默认训练配置。")
        return self.train(config_path=None)

    def _create_default_train_config(self, **kwargs) -> str:
        """
        创建默认训练配置
        当没有提供配置文件时，使用此方法生成默认配置
        所有参数都有合理的默认值，适合Llama-3.2-1B模型的LoRA微调
        """
        default_config = {
            # === 模型配置 ===
            "### model": None,
            "model_name_or_path": kwargs.get("model_name_or_path", "models/Qwen3-0.6B"),  # 本地模型路径
            "trust_remote_code": True,  # 信任远程代码，允许加载自定义模型
            
            # === 微调方法配置 ===
            "### method": None,
            "stage": kwargs.get("stage", "sft"),  # 监督微调阶段
            "do_train": True,  # 启用训练模式
            "finetuning_type": kwargs.get("finetuning_type", "lora"),  # 使用LoRA微调
            "lora_rank": kwargs.get("lora_rank", 8),  # LoRA秩，控制参数量
            "lora_target": kwargs.get("lora_target", "q_proj,v_proj"),  # LoRA作用模块，减少参数量
            
            # === 数据集配置 ===
            "### dataset": None,
            "dataset": kwargs.get("dataset", "identity,alpaca_en_demo"),  # 默认数据集
            "template": kwargs.get("template", "qwen3"),  # 对话模板
            "cutoff_len": kwargs.get("cutoff_len", 2048),  # 文本截断长度
            "max_samples": kwargs.get("max_samples", 1000),  # 最大样本数
            "overwrite_cache": True,  # 覆盖缓存
            "preprocessing_num_workers": 1,  # Windows多进程问题，设置为1避免pickle错误
            "dataloader_num_workers": 0,     # Windows多进程问题，设置为0避免spawn错误
            
            # === 输出配置 ===
            "### output": None,
            "output_dir": kwargs.get("output_dir", str(LOCAL_RUN_DIR / "llama3.2-1b-lora-sft")),  # 输出目录
            "logging_steps": kwargs.get("logging_steps", 10),  # 日志记录步数
            "save_steps": kwargs.get("save_steps", 500),  # 模型保存步数
            "plot_loss": True,  # 绘制损失曲线
            "overwrite_output_dir": True,  # 覆盖输出目录
            "save_only_model": False,  # 保存完整训练状态
            "report_to": kwargs.get("report_to", "none"),  # 不向外部平台报告
            
            # === 训练参数配置 ===
            "### train": None,
            "per_device_train_batch_size": kwargs.get("per_device_train_batch_size", 1),  # 批次大小
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 8),  # 梯度累积
            "learning_rate": kwargs.get("learning_rate", 1.0e-4),  # 学习率
            "num_train_epochs": kwargs.get("num_train_epochs", 3.0),  # 训练轮数
            "lr_scheduler_type": kwargs.get("lr_scheduler_type", "cosine"),  # 学习率调度器
            "warmup_ratio": kwargs.get("warmup_ratio", 0.1),  # 预热比例
            "bf16": kwargs.get("bf16", True),  # 使用bfloat16精度
            "ddp_timeout": 180000000,  # 分布式训练超时
            "resume_from_checkpoint": None  # 不从检查点恢复
        }
        config_path = self.output_dir / "default_train_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            for key, value in default_config.items():
                if key.startswith("###"):
                    f.write(f"\n{key}\n")
                else:
                    f.write(f"{key}: {value}\n")
        return str(config_path)

    def chat(self, model_path: str, adapter_path: Optional[str] = None,
             template: str = "qwen3", **kwargs) -> int:
        logger.info(f"启动CLI聊天，模型: {model_path}")
        # 直接以源码方式创建 ChatModel 并进入交互
        args = {
            "model_name_or_path": model_path,
            "template": template,
            "trust_remote_code": True,
        }
        
        # 尝试使用vllm，如果不可用则回退到huggingface
        try:
            import vllm
            args["infer_backend"] = "vllm"
            logger.info("使用vLLM推理引擎")
        except ImportError:
            args["infer_backend"] = "huggingface"
            logger.info("vLLM不可用，使用HuggingFace推理引擎")
        if adapter_path:
            args["adapter_name_or_path"] = adapter_path
        args.update(kwargs)
        return self._run_chat_with_args(args)

    def _run_chat_with_args(self, args: dict) -> int:
        try:
            chat_model = ChatModel(args=args)
        except Exception as e:
            logger.error(f"初始化 ChatModel 失败：{e}")
            return 1
        print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
        messages: list[dict[str, str]] = []
        while True:
            try:
                query = input("\nUser: ")
            except Exception:
                raise
            if query.strip() == "exit":
                break
            if query.strip() == "clear":
                messages = []
                print("History has been removed.")
                continue
            messages.append({"role": "user", "content": query})
            print("Assistant: ", end="", flush=True)
            response = ""
            for new_text in chat_model.stream_chat(messages):
                print(new_text, end="", flush=True)
                response += new_text
            print()
            messages.append({"role": "assistant", "content": response})
        return 0

    def webui(self, **kwargs) -> int:
        logger.info("启动Web UI...")
        run_web_ui()
        return 0

    def webchat(self, model_path: str, adapter_path: Optional[str] = None,
                template: str = "qwen3", **kwargs) -> int:
        logger.info(f"启动Web聊天，模型: {model_path}")
        config = {
            "model_name_or_path": model_path,
            "template": template,
            "trust_remote_code": True,
        }
        
        # 尝试使用vllm，如果不可用则回退到huggingface
        try:
            import vllm
            config["infer_backend"] = "vllm"
            logger.info("使用vLLM推理引擎")
        except ImportError:
            config["infer_backend"] = "huggingface"
            logger.info("vLLM不可用，使用HuggingFace推理引擎")
        if adapter_path:
            config["adapter_name_or_path"] = adapter_path
        config.update(kwargs)
        config_path = self.output_dir / "webchat_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        run_web_demo()
        return 0

    def api(self, model_path: str, adapter_path: Optional[str] = None,
            template: str = "qwen3", port: int = 8000, host: str = "0.0.0.0",
            **kwargs) -> int:
        logger.info(f"启动API服务，模型: {model_path}, 端口: {port}")
        config = {
            "model_name_or_path": model_path,
            "template": template,
            "trust_remote_code": True,
        }
        
        # 尝试使用vllm，如果不可用则回退到huggingface
        try:
            import vllm
            config["infer_backend"] = "vllm"
            logger.info("使用vLLM推理引擎")
        except ImportError:
            config["infer_backend"] = "huggingface"
            logger.info("vLLM不可用，使用HuggingFace推理引擎")
        if adapter_path:
            config["adapter_name_or_path"] = adapter_path
        config.update(kwargs)
        config_path = self.output_dir / "api_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        if "host" in kwargs:
            os.environ["API_HOST"] = str(kwargs.get("host"))
        if "port" in kwargs:
            os.environ["API_PORT"] = str(kwargs.get("port"))
        run_api()
        return 0

    def export(self, model_path: str, adapter_path: str,
               export_dir: str, template: str = "qwen3", **kwargs) -> int:
        logger.info(f"导出模型，基础模型: {model_path}, 适配器: {adapter_path}")
        config = {
            "model_name_or_path": model_path,
            "adapter_name_or_path": adapter_path,
            "template": template,
            "export_dir": export_dir,
            "trust_remote_code": True
        }
        config.update(kwargs)
        config_path = self.output_dir / "export_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        export_model(args=config)
        return 0

    def eval(self, config_path: str, **kwargs) -> int:
        logger.info(f"开始评估，配置文件: {config_path}")
        # 直接源码：Evaluator(args=dict) 调用
        args = self._load_task_args(config_path, kwargs)
        Evaluator(args=args).eval()
        return 0


def main():
    # 零命令行参数：直接从运行配置文件启动
    if len(sys.argv) == 1:
        runner = LLaMAFactoryRunner()
        rc = runner.run_from_file()
        sys.exit(rc)
        return

    parser = argparse.ArgumentParser(
        description="LLaMA-Factory本地执行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    train_parser = subparsers.add_subparser('train', help='训练模型') if hasattr(subparsers, 'add_subparser') else subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', default=str(LOCAL_RUN_DIR / 'qwen3_lora_sft.yaml'), help='配置文件路径（可选，不指定则使用默认配置）')
    train_parser.add_argument('--model', default='models/Llama-3.2-1B-Instruct', help='模型路径')
    train_parser.add_argument('--dataset', default='identity,alpaca_en_demo', help='数据集名称')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    train_parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    train_parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    train_parser.add_argument('--output-dir', default=str(LOCAL_RUN_DIR / 'llama3.2-1b-lora-sft'), help='输出目录')
    train_parser.add_argument('--finetuning-type', choices=['lora', 'qlora', 'full'], default='lora', help='微调类型')
    train_parser.add_argument('--lora-rank', type=int, default=8, help='LoRA秩')
    train_parser.add_argument('--max-samples', type=int, default=1000, help='最大样本数')

    chat_parser = subparsers.add_parser('chat', help='CLI聊天')
    chat_parser.add_argument('--model', default='models/Llama-3.2-1B-Instruct', help='模型路径')
    chat_parser.add_argument('--adapter', help='适配器路径')
    chat_parser.add_argument('--template', default='llama3', help='模板名称')

    webchat_parser = subparsers.add_parser('webchat', help='Web聊天')
    webchat_parser.add_argument('--model', default='models/Llama-3.2-1B-Instruct', help='模型路径')
    webchat_parser.add_argument('--adapter', help='适配器路径')
    webchat_parser.add_argument('--template', default='llama3', help='模板名称')

    api_parser = subparsers.add_parser('api', help='启动API服务')
    api_parser.add_argument('--model', default='models/Llama-3.2-1B-Instruct', help='模型路径')
    api_parser.add_argument('--adapter', help='适配器路径')
    api_parser.add_argument('--template', default='llama3', help='模板名称')
    api_parser.add_argument('--port', type=int, default=8000, help='端口号')
    api_parser.add_argument('--host', default='0.0.0.0', help='主机地址')

    webui_parser = subparsers.add_parser('webui', help='启动Web UI')
    webui_parser.add_argument('--port', type=int, default=7860, help='端口号')
    webui_parser.add_argument('--host', default='0.0.0.0', help='主机地址')

    export_parser = subparsers.add_parser('export', help='导出模型')
    export_parser.add_argument('--model', required=True, help='基础模型路径')
    export_parser.add_argument('--adapter', required=True, help='适配器路径')
    export_parser.add_argument('--export-dir', required=True, help='导出目录')
    export_parser.add_argument('--template', default='llama3', help='模板名称')

    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('--config', required=True, help='配置文件路径')

    args = parser.parse_args()
    runner = LLaMAFactoryRunner()

    try:
        if args.command == 'train':
            kwargs = {}
            if getattr(args, 'model', None):
                kwargs['model_name_or_path'] = args.model
            if getattr(args, 'dataset', None):
                kwargs['dataset'] = args.dataset
            if getattr(args, 'learning_rate', None) is not None:
                kwargs['learning_rate'] = args.learning_rate
            if getattr(args, 'batch_size', None) is not None:
                kwargs['per_device_train_batch_size'] = args.batch_size
            if getattr(args, 'epochs', None) is not None:
                kwargs['num_train_epochs'] = args.epochs
            if getattr(args, 'output_dir', None):
                kwargs['output_dir'] = args.output_dir
            if getattr(args, 'finetuning_type', None):
                kwargs['finetuning_type'] = args.finetuning_type
            if getattr(args, 'lora_rank', None) is not None:
                kwargs['lora_rank'] = args.lora_rank
            if getattr(args, 'max_samples', None) is not None:
                kwargs['max_samples'] = args.max_samples
            return_code = runner.train(args.config, **kwargs)
        elif args.command == 'chat':
            return_code = runner.chat(
                model_path=args.model,
                adapter_path=args.adapter,
                template=args.template
            )
        elif args.command == 'webchat':
            return_code = runner.webchat(
                model_path=args.model,
                adapter_path=args.adapter,
                template=args.template
            )
        elif args.command == 'api':
            return_code = runner.api(
                model_path=args.model,
                adapter_path=args.adapter,
                template=args.template,
                port=args.port,
                host=args.host
            )
        elif args.command == 'webui':
            return_code = runner.webui(port=args.port, host=args.host)
        elif args.command == 'export':
            return_code = runner.export(
                model_path=args.model,
                adapter_path=args.adapter,
                export_dir=args.export_dir,
                template=args.template
            )
        elif args.command == 'eval':
            return_code = runner.eval(args.config)
        else:
            logger.error(f"未知命令: {args.command}")
            return_code = 1
        sys.exit(return_code)
    except KeyboardInterrupt:
        logger.info("用户中断执行")
        sys.exit(0)
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


