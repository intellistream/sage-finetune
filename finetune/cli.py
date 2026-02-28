#!/usr/bin/env python3
"""
Finetune CLI - Command Handlers
CLI 命令处理器（精简版，调用模块化的核心逻辑）

本文件只负责：
1. 定义 CLI 命令和参数
2. 处理用户交互
3. 调用核心模块完成实际工作
"""

import json
import shutil
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from sage.libs.finetune.core import generate_training_config, prepare_training_data
from sage.libs.finetune.models import TASK_NAMES, FinetuneTask
from sage.libs.finetune.service import (
    merge_lora_weights,
    start_training,
)
from sage.libs.finetune.utils import (
    check_training_dependencies,
    get_finetune_output_dir,
    get_sage_root,
    show_install_instructions,
)

app = typer.Typer(
    name="finetune",
    help="🎓 大模型微调工具 - 支持代码理解、对话、指令等多种场景",
)

console = Console()


@app.command("start")
def start_finetune(
    task: str | None = typer.Option(None, "--task", "-t", help="任务类型"),
    model: str | None = typer.Option(None, "--model", "-m", help="基础模型"),
    data: str | None = typer.Option(None, "--data", "-d", help="数据文件"),
    output: str | None = typer.Option(None, "--output", "-o", help="输出目录"),
    framework: str = typer.Option("llama-factory", "--framework", "-f"),
    format: str = typer.Option("alpaca", "--format"),
    auto: bool = typer.Option(False, "--auto", help="自动模式"),
    skip_install: bool = typer.Option(False, "--skip-install"),
):
    """🎓 启动交互式微调流程"""
    console.print(Panel.fit("[bold cyan]🎓 SAGE大模型微调向导[/bold cyan]", border_style="cyan"))

    # 选择任务类型
    if not task and not auto:
        task = _select_task()
    elif not task:
        task = "code"

    task_type = FinetuneTask(task)
    console.print(f"\n✅ 任务: [green]{TASK_NAMES[task_type]}[/green]\n")

    # 检查依赖
    if not skip_install and not check_training_dependencies():
        show_install_instructions()
        console.print("\n[red]❌ 请先安装依赖或使用 --skip-install[/red]")
        raise typer.Exit(1)

    # 选择模型
    if not model:
        model = _select_model(task_type, auto)
    console.print(f"✅ 模型: [green]{model}[/green]\n")

    # 处理数据源
    root_dir, custom_data_path = _handle_data_source(task_type, data, auto)

    # 设置输出目录
    if not output:
        output_dir = get_finetune_output_dir() / task_type.value
    else:
        output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"✅ 输出: [green]{output_dir}[/green]\n")

    # 生成训练数据
    console.print("[bold]📝 生成训练数据...[/bold]")
    dataset_path = prepare_training_data(
        task_type, root_dir or Path.cwd(), output_dir, format, custom_data_path
    )

    # 生成配置
    console.print("\n[bold]⚙️  生成配置...[/bold]")
    config_path = generate_training_config(model, dataset_path, output_dir, framework)
    console.print(f"✅ 配置: [cyan]{config_path}[/cyan]\n")

    # 保存元信息
    _save_meta(
        output_dir,
        task_type,
        model,
        framework,
        format,
        dataset_path,
        config_path,
        root_dir,
        custom_data_path,
    )

    # 显示下一步
    _show_next_steps(output_dir, config_path)

    # 询问是否立即训练
    if not auto and Confirm.ask("\n是否立即启动训练?", default=False):
        console.print("\n[bold]🚀 启动训练...[/bold]")
        start_training(config_path, use_native=True)


@app.command("run")
def run_training(
    config: str = typer.Argument(..., help="配置文件或输出目录"),
    use_native: bool = typer.Option(True, "--use-native/--use-llamafactory"),
):
    """🚀 运行微调训练"""
    config_path = Path(config)

    if config_path.is_dir():
        possible_configs = list(config_path.glob("*.json"))
        if possible_configs:
            config_path = possible_configs[0]

    if not config_path.exists():
        console.print(f"[red]❌ 配置不存在: {config}[/red]")
        raise typer.Exit(1)

    console.print("[bold]🚀 启动训练[/bold]")
    console.print(f"配置: [cyan]{config_path}[/cyan]\n")

    start_training(config_path, use_native)


@app.command("list")
def list_outputs(directory: str | None = typer.Option(None, "--dir", "-d")):
    """📋 列出所有微调输出"""
    output_dir = Path(directory) if directory else get_finetune_output_dir()

    if not output_dir.exists():
        console.print(f"[yellow]⚠️  目录不存在: {output_dir}[/yellow]")
        return

    meta_files = list(output_dir.glob("**/finetune_meta.json"))

    if not meta_files:
        console.print("[yellow]📭 没有找到微调任务[/yellow]")
        return

    table = Table(title=f"微调任务列表 ({len(meta_files)} 个)")
    table.add_column("序号", style="cyan")
    table.add_column("模型", style="green")
    table.add_column("任务", style="yellow")
    table.add_column("输出目录", style="blue")

    for i, meta_file in enumerate(meta_files, 1):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            table.add_row(
                str(i),
                meta.get("model", "N/A"),
                meta.get("task_type", "N/A"),
                Path(meta.get("output_dir", "")).name,
            )
        except Exception as e:
            console.print(f"[red]读取失败: {e}[/red]")

    console.print(table)


@app.command("clean")
def clean_outputs(
    directory: str | None = typer.Option(None, "--dir", "-d"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """🧹 清理微调输出"""
    output_dir = Path(directory) if directory else get_finetune_output_dir()

    if not output_dir.exists():
        console.print("[yellow]⚠️  目录不存在[/yellow]")
        return

    console.print(f"将删除: [red]{output_dir}[/red]")

    if not force and not Confirm.ask("确认删除?", default=False):
        console.print("[yellow]已取消[/yellow]")
        return

    shutil.rmtree(output_dir)
    console.print("[green]✅ 已删除[/green]")


@app.command("quickstart")
def quickstart(task: str = typer.Argument("code", help="任务类型")):
    """🚀 快速开始"""
    console.print(Panel.fit(f"[bold cyan]🚀 快速开始 - {task}[/bold cyan]", border_style="cyan"))

    if task == "code":
        console.print("\n[bold green]📚 SAGE代码理解快速微调[/bold green]")
        console.print("默认配置: Qwen/Qwen2.5-Coder-7B-Instruct\n")

        if Confirm.ask("确认开始?", default=True):
            start_finetune(
                task="code",
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                framework="llama-factory",
                format="alpaca",
                auto=True,
                output=None,
                data=None,
                skip_install=True,
            )
    else:
        console.print(f"\n[yellow]⚠️  {task}任务需要数据文件[/yellow]")
        console.print(f"使用: [cyan]sage finetune start --task {task} --data <file>[/cyan]")


@app.command("merge")
def merge_lora(
    model_name: str = typer.Argument(..., help="模型名称或路径"),
    output: str | None = typer.Option(None, "--output", "-o"),
):
    """🔀 合并 LoRA 权重"""
    console.print("[bold]🔀 合并 LoRA 权重[/bold]\n")

    checkpoint_path, base_model = _find_checkpoint(model_name)

    if not output:
        output_path = checkpoint_path.parent.parent / "merged_model"
    else:
        output_path = Path(output)

    output_path.mkdir(parents=True, exist_ok=True)

    if merge_lora_weights(checkpoint_path, base_model, output_path):
        console.print("\n[bold]💡 使用:[/bold]")
        console.print(f"[cyan]sage finetune chat {model_name}[/cyan]")


@app.command("serve")
def serve_model(
    model_name: str = typer.Argument(..., help="模型名称"),
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host"),
    daemon: bool = typer.Option(False, "--daemon", "-d"),
    gpu_memory_utilization: float = typer.Option(0.9, "--gpu-util"),
):
    """🚀 启动模型服务

    Deprecated: 此命令已废弃，请使用统一的 sageLLM 服务：
        sage llm serve --model <model_path>
    """
    console.print("[yellow]⚠️  此命令已废弃[/yellow]\n")
    console.print("请使用统一的 sageLLM 服务命令：\n")

    model_path, use_lora, lora_path = _find_model_for_serving(model_name)

    cmd_parts = [
        "sage",
        "llm",
        "serve",
        "--model",
        str(model_path),
        "--port",
        str(port),
        "--host",
        host,
    ]

    if not daemon:
        cmd_parts.append("--blocking")

    if use_lora and lora_path:
        console.print(
            "[yellow]注意: LoRA 支持需要手动配置，请参考 sage llm serve --help[/yellow]\n"
        )

    console.print(f"[cyan]建议命令: {' '.join(cmd_parts)}[/cyan]\n")
    console.print("是否执行? [Y/n]: ", end="")

    if Confirm.ask("", default=True):
        subprocess.run(cmd_parts)


@app.command("chat")
def auto_chat(
    model_name: str = typer.Argument("sage_code_expert", help="模型名称"),
    port: int = typer.Option(8000, "--port", "-p"),
):
    """💬 使用微调模型聊天（重定向到 sage chat）"""
    console.print("[cyan]ℹ️  重定向到 sage chat...[/cyan]\n")

    try:
        subprocess.run(
            [
                "sage",
                "chat",
                "--backend",
                "finetune",
                "--finetune-model",
                model_name,
                "--finetune-port",
                str(port),
            ]
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 退出[/yellow]")
    except Exception as exc:
        console.print(f"[red]❌ 失败: {exc}[/red]")
        raise typer.Exit(1)


@app.command("examples")
def show_examples():
    """📚 显示使用示例"""
    console.print(Panel.fit("[bold cyan]📚 使用示例[/bold cyan]", border_style="cyan"))

    examples = [
        ("SAGE代码理解", "sage finetune quickstart code"),
        ("问答微调", "sage finetune start --task qa --data qa.json"),
        ("指令微调", "sage finetune start --task instruction --data inst.json"),
        ("运行训练", "sage finetune run ~/.sage/finetune_output/code"),
        ("合并模型", "sage finetune merge code"),
        ("启动服务", "sage finetune serve code --port 8000"),
        ("聊天测试", "sage finetune chat code"),
    ]

    table = Table(show_header=True)
    table.add_column("场景", style="cyan")
    table.add_column("命令", style="green")

    for scene, cmd in examples:
        table.add_row(scene, cmd)

    console.print("\n", table, "\n")


# ========== 辅助函数 ==========


def _select_task() -> str:
    """交互式选择任务类型"""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("序号", width=6)
    table.add_column("类型", width=15)
    table.add_column("说明")

    tasks = [
        ("1", "code", "代码理解微调"),
        ("2", "qa", "问答对微调"),
        ("3", "instruction", "指令微调"),
        ("4", "chat", "对话微调"),
        ("5", "custom", "自定义数据集"),
    ]

    for num, typ, desc in tasks:
        table.add_row(num, typ, desc)

    console.print(table)
    choice = IntPrompt.ask("选择任务类型", default=1)

    task_map = {1: "code", 2: "qa", 3: "instruction", 4: "chat", 5: "custom"}
    return task_map.get(choice, "code")


def _select_model(task_type: FinetuneTask, auto: bool) -> str:
    """选择基础模型"""
    if auto:
        return (
            "Qwen/Qwen2.5-Coder-7B-Instruct"
            if task_type == FinetuneTask.CODE_UNDERSTANDING
            else "Qwen/Qwen2.5-7B-Instruct"
        )

    if task_type == FinetuneTask.CODE_UNDERSTANDING:
        console.print("推荐: Qwen/Qwen2.5-Coder-7B-Instruct (代码专精)")
        default = "Qwen/Qwen2.5-Coder-7B-Instruct"
    else:
        console.print("推荐: Qwen/Qwen2.5-7B-Instruct (通用)")
        default = "Qwen/Qwen2.5-7B-Instruct"

    return Prompt.ask("模型名称", default=default)


def _handle_data_source(task_type: FinetuneTask, data: str | None, auto: bool):
    """处理数据源"""
    if task_type == FinetuneTask.CODE_UNDERSTANDING:
        sage_root = get_sage_root()
        if auto or Confirm.ask(f"使用SAGE代码库? {sage_root}", default=True):
            return sage_root, None
        custom_path = Prompt.ask("代码库路径")
        return Path(custom_path), None
    else:
        if not data:
            data = Prompt.ask("数据文件路径")
        custom_data_path = Path(data)
        if not custom_data_path.exists():
            console.print(f"[red]❌ 文件不存在: {data}[/red]")
            raise typer.Exit(1)
        return None, custom_data_path


def _save_meta(
    output_dir,
    task_type,
    model,
    framework,
    format,
    dataset_path,
    config_path,
    root_dir,
    custom_data_path,
):
    """保存元信息"""
    meta = {
        "task_type": task_type.value,
        "model": model,
        "framework": framework,
        "format": format,
        "dataset": str(dataset_path),
        "config": str(config_path),
        "output_dir": str(output_dir),
    }

    if root_dir:
        meta["code_root"] = str(root_dir)
    if custom_data_path:
        meta["data_file"] = str(custom_data_path)

    meta_path = output_dir / "finetune_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    console.print(f"💾 元信息: [cyan]{meta_path}[/cyan]")


def _show_next_steps(output_dir, config_path):
    """显示下一步操作"""
    console.print(
        Panel.fit(
            "[bold green]✅ 准备完成！[/bold green]\n\n"
            "[bold]🚀 启动训练:[/bold]\n"
            f"[cyan]sage finetune run {output_dir}[/cyan]\n\n"
            f"或: [cyan]python -m sage.libs.finetune.trainer {output_dir}[/cyan]",
            border_style="green",
        )
    )


def _find_checkpoint(model_name: str):
    """查找 checkpoint 和基础模型"""
    checkpoint_path = Path(model_name)

    if not checkpoint_path.exists():
        output_dir = get_finetune_output_dir()
        checkpoint_dir = output_dir / model_name / "checkpoints"

        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                console.print(f"✅ Checkpoint: [cyan]{checkpoint_path}[/cyan]\n")
            else:
                console.print("[red]❌ 未找到checkpoint[/red]")
                raise typer.Exit(1)
        else:
            console.print(f"[red]❌ 未找到: {model_name}[/red]")
            raise typer.Exit(1)

    # 读取meta
    meta_file = checkpoint_path.parent.parent / "finetune_meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        base_model = meta.get("model", "")
        console.print(f"📦 基础模型: [green]{base_model}[/green]\n")
        return checkpoint_path, base_model
    else:
        console.print("[yellow]⚠️  未找到meta，需手动指定基础模型[/yellow]")
        base_model = Prompt.ask("基础模型名称")
        return checkpoint_path, base_model


def _find_model_for_serving(model_name: str):
    """查找用于服务的模型路径"""
    output_dir = get_finetune_output_dir()

    # 优先查找合并模型
    merged_path = output_dir / model_name / "merged_model"
    checkpoint_path = output_dir / model_name / "checkpoints"

    if merged_path.exists():
        console.print(f"✅ 合并模型: [cyan]{merged_path}[/cyan]\n")
        return merged_path, False, None
    elif checkpoint_path.exists():
        # 使用 LoRA 模式
        checkpoints = sorted(checkpoint_path.glob("checkpoint-*"))
        if checkpoints:
            lora_path = checkpoints[-1]
            meta_file = checkpoint_path.parent / "finetune_meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                base_model = meta.get("model", "")
                console.print(f"✅ LoRA: [cyan]{base_model}[/cyan] + [cyan]{lora_path}[/cyan]\n")
                return Path(base_model), True, lora_path

    console.print(f"[red]❌ 未找到: {model_name}[/red]")
    console.print("[yellow]提示: 先运行 sage finetune merge[/yellow]")
    raise typer.Exit(1)
