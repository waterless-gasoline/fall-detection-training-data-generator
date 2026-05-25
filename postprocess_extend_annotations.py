from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ALLOWED_DELTAS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复制标注目录并随机延长每个跌倒事件的结束时间")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(r"D:\IPC\IPC_video_data_annotation\跌倒"),
        help="源标注目录",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(r"D:\IPC\IPC_video_data_annotation\跌倒_v2"),
        help="目标输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="可选随机种子，便于复现结果",
    )
    return parser.parse_args()


def format_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def extend_annotation_line(line: str, rng: random.Random) -> tuple[str, bool]:
    stripped = line.strip()
    if not stripped or ";" not in stripped:
        return line, False

    prefix, time_part = stripped.split(";", 1)
    if "-" not in time_part:
        return line, False

    start_text, end_text = time_part.split("-", 1)

    try:
        start = float(start_text)
        end = float(end_text)
    except ValueError:
        return line, False

    delta = rng.choice(ALLOWED_DELTAS)
    new_end = end + delta
    new_line = f"{prefix};{format_number(start)}-{format_number(new_end)}"
    if line.endswith("\n"):
        new_line += "\n"
    return new_line, True


def process_annotation_file(file_path: Path, rng: random.Random) -> int:
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_lines = []
    changed_events = 0

    for line in lines:
        new_line, changed = extend_annotation_line(line, rng)
        updated_lines.append(new_line)
        if changed:
            changed_events += 1

    file_path.write_text("".join(updated_lines), encoding="utf-8")
    return changed_events


def main() -> None:
    args = parse_args()

    if not args.src.exists() or not args.src.is_dir():
        raise FileNotFoundError(f"源目录不存在: {args.src}")
    if args.dst.exists():
        raise FileExistsError(f"目标目录已存在，请先删除或更换输出目录: {args.dst}")

    rng = random.Random(args.seed)

    shutil.copytree(args.src, args.dst)

    annotation_files = sorted(args.dst.rglob("annotation.txt"))
    changed_events = 0
    for annotation_file in annotation_files:
        changed_events += process_annotation_file(annotation_file, rng)

    print(f"已复制目录: {args.src} -> {args.dst}")
    print(f"处理 annotation.txt 数量: {len(annotation_files)}")
    print(f"延长跌倒事件数量: {changed_events}")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")


if __name__ == "__main__":
    main()
