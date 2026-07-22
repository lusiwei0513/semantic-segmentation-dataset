from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


@dataclass
class SplitResult:
    separator_start: int
    separator_end: int
    background_rgb: tuple[int, int, int]


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(output_path, quality=92)
    else:
        image.save(output_path)


def estimate_background(array: np.ndarray) -> np.ndarray:
    """用四个角的像素估计背景颜色，适合白色或近白色背景。"""
    height, width, _ = array.shape
    patch = max(5, min(height, width) // 40)

    corners = np.concatenate(
        [
            array[:patch, :patch].reshape(-1, 3),
            array[:patch, -patch:].reshape(-1, 3),
            array[-patch:, :patch].reshape(-1, 3),
            array[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(corners, axis=0)


def foreground_mask(
    image: Image.Image,
    color_tolerance: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    根据像素与背景颜色的差异生成前景掩膜。
    color_tolerance 越小越敏感，越大越容易忽略浅色车身边缘。
    """
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    background = estimate_background(array)
    difference = np.max(np.abs(array - background), axis=2)
    mask = difference > color_tolerance

    background_rgb = tuple(int(round(value)) for value in background)
    return mask, background_rgb


def find_separator(
    image: Image.Image,
    color_tolerance: int,
    search_start_ratio: float,
    search_end_ratio: float,
    max_foreground_ratio: float,
    minimum_separator_width: int,
) -> SplitResult:
    """
    在图像左侧指定范围中寻找一条连续的近空白竖向分隔带。

    返回分隔带起点与终点，而不是只返回一个固定 x 坐标。
    后续会把整条白色分隔带删除，避免切到列车主体。
    """
    mask, background_rgb = foreground_mask(image, color_tolerance)
    height, width = mask.shape

    search_start = max(1, round(width * search_start_ratio))
    search_end = min(width - 1, round(width * search_end_ratio))

    if search_start >= search_end:
        raise ValueError("分隔带搜索范围无效。")

    column_foreground = mask.sum(axis=0)
    max_foreground_pixels = max(
        2,
        round(height * max_foreground_ratio),
    )

    candidate = (
        column_foreground[search_start:search_end]
        <= max_foreground_pixels
    )

    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for offset, is_blank in enumerate(candidate):
        x = search_start + offset

        if is_blank and run_start is None:
            run_start = x
        elif not is_blank and run_start is not None:
            run_end = x - 1
            if run_end - run_start + 1 >= minimum_separator_width:
                runs.append((run_start, run_end))
            run_start = None

    if run_start is not None:
        run_end = search_end - 1
        if run_end - run_start + 1 >= minimum_separator_width:
            runs.append((run_start, run_end))

    valid_runs: list[tuple[int, int]] = []

    for start, end in runs:
        # 分隔带左右两侧都必须确实存在前景，避免把图像外侧白边误判为分隔带。
        left_foreground = int(mask[:, :start].sum())
        right_foreground = int(mask[:, end + 1 :].sum())

        minimum_content = max(20, round(height * width * 0.0005))

        if (
            left_foreground >= minimum_content
            and right_foreground >= minimum_content
        ):
            valid_runs.append((start, end))

    if not valid_runs:
        raise ValueError(
            "未找到可靠的正视图/侧视图白色分隔带。"
            "该图片可能不是组合图、正视图不在左侧，"
            "或背景并非白色。"
        )

    # 优先选择最宽的空白带；宽度相同时选择更靠近左侧预期位置的空白带。
    separator_start, separator_end = max(
        valid_runs,
        key=lambda item: (
            item[1] - item[0] + 1,
            -item[0],
        ),
    )

    return SplitResult(
        separator_start=separator_start,
        separator_end=separator_end,
        background_rgb=background_rgb,
    )


def trim_background(
    image: Image.Image,
    color_tolerance: int,
    padding: int,
) -> Image.Image:
    """去除切片四周的空白边，并保留少量 padding。"""
    mask, _ = foreground_mask(image, color_tolerance)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return image

    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(image.width, int(xs.max()) + 1 + padding)
    bottom = min(image.height, int(ys.max()) + 1 + padding)

    return image.crop((left, top, right, bottom))


def create_preview(
    image: Image.Image,
    separator_start: int,
    separator_end: int,
    output_path: Path,
) -> None:
    """生成带两条竖线的预览图，便于人工核对自动切割位置。"""
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)

    line_width = max(2, image.width // 800)
    draw.line(
        [(separator_start, 0), (separator_start, image.height - 1)],
        fill=(255, 0, 0),
        width=line_width,
    )
    draw.line(
        [(separator_end, 0), (separator_end, image.height - 1)],
        fill=(0, 0, 255),
        width=line_width,
    )

    save_image(preview, output_path)


def process_image(
    source_path: Path,
    front_directory: Path,
    side_directory: Path,
    preview_directory: Path,
    front_position: str,
    min_aspect_ratio: float,
    color_tolerance: int,
    search_start_ratio: float,
    search_end_ratio: float,
    max_foreground_ratio: float,
    minimum_separator_width: int,
    trim: bool,
    trim_padding: int,
    overwrite: bool,
) -> dict[str, object]:
    with Image.open(source_path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")
        image.load()

    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio < min_aspect_ratio:
        raise ValueError(
            f"宽高比仅为 {aspect_ratio:.2f}，不像正侧视组合图。"
            f"当前最低要求为 {min_aspect_ratio:.2f}。"
        )

    result = find_separator(
        image=image,
        color_tolerance=color_tolerance,
        search_start_ratio=search_start_ratio,
        search_end_ratio=search_end_ratio,
        max_foreground_ratio=max_foreground_ratio,
        minimum_separator_width=minimum_separator_width,
    )

    # 删除完整白色分隔带：
    # 左图截止到 separator_start，右图从 separator_end + 1 开始。
    separator_middle = (
        result.separator_start + result.separator_end
    ) // 2

    left_image = image.crop(
        (0, 0, separator_middle, height)
    )

    right_image = image.crop(
        (separator_middle, 0, width, height)
    )

    if front_position == "left":
        front_image = left_image
        side_image = right_image
    else:
        front_image = right_image
        side_image = left_image

    if trim:
        front_image = trim_background(
            front_image,
            color_tolerance=color_tolerance,
            padding=trim_padding,
        )
        side_image = trim_background(
            side_image,
            color_tolerance=color_tolerance,
            padding=trim_padding,
        )

    suffix = source_path.suffix.lower()
    front_path = front_directory / f"{source_path.stem}_front{suffix}"
    side_path = side_directory / f"{source_path.stem}_side{suffix}"
    preview_path = preview_directory / f"{source_path.stem}_preview.jpg"

    if not overwrite and (front_path.exists() or side_path.exists()):
        raise FileExistsError(
            "输出文件已存在；需要覆盖时添加 --overwrite。"
        )

    save_image(front_image, front_path)
    save_image(side_image, side_path)
    # create_preview(
    #     image=image,
    #     separator_start=result.separator_start,
    #     separator_end=result.separator_end,
    #     output_path=preview_path,
    # )

    return {
        "source": source_path.name,
        "status": "success",
        "source_width": width,
        "source_height": height,
        "aspect_ratio": f"{aspect_ratio:.4f}",
        "separator_start": result.separator_start,
        "separator_end": result.separator_end,
        "front_width": front_image.width,
        "front_height": front_image.height,
        "side_width": side_image.width,
        "side_height": side_image.height,
        "background_rgb": result.background_rgb,
        "message": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "自动寻找白色竖向分隔带，批量拆分列车正视图和侧视图。"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="组合图片所在文件夹",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="输出根目录",
    )
    parser.add_argument(
        "--front-position",
        choices=["left", "right"],
        default="left",
        help="正视图位于组合图左侧还是右侧，默认 left",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=2.2,
        help=(
            "低于该宽高比的图片会被跳过，防止把普通单图误切。"
            "默认 2.2。"
        ),
    )
    parser.add_argument(
        "--color-tolerance",
        type=int,
        default=18,
        help=(
            "像素与背景颜色差异阈值，默认18。"
            "浅色车体被忽略时可减小到12；噪声过多时可增大到25。"
        ),
    )
    parser.add_argument(
        "--search-start-ratio",
        type=float,
        default=0.05,
        help="分隔带搜索起点占图片宽度的比例，默认0.05",
    )
    parser.add_argument(
        "--search-end-ratio",
        type=float,
        default=0.45,
        help="分隔带搜索终点占图片宽度的比例，默认0.45",
    )
    parser.add_argument(
        "--max-foreground-ratio",
        type=float,
        default=0.01,
        help=(
            "候选空白列允许的前景像素占图片高度比例，默认0.01"
        ),
    )
    parser.add_argument(
        "--minimum-separator-width",
        type=int,
        default=8,
        help="白色分隔带最小宽度，默认8像素",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="不自动去除两个切片四周的白边",
    )
    parser.add_argument(
        "--trim-padding",
        type=int,
        default=10,
        help="自动去白边后保留的边距，默认10像素",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的拆分结果",
    )

    args = parser.parse_args()

    input_directory = args.input.expanduser().resolve()
    output_directory = args.output.expanduser().resolve()

    if not input_directory.is_dir():
        raise NotADirectoryError(
            f"输入文件夹不存在或不是目录：{input_directory}"
        )

    front_directory = output_directory / "front_images"
    side_directory = output_directory / "side_images"
    preview_directory = output_directory / "previews"

    front_directory.mkdir(parents=True, exist_ok=True)
    side_directory.mkdir(parents=True, exist_ok=True)
    preview_directory.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in input_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(f"没有找到可处理图片：{input_directory}")
        return

    report_rows: list[dict[str, object]] = []
    success_count = 0
    skipped_count = 0

    for source_path in image_paths:
        try:
            row = process_image(
                source_path=source_path,
                front_directory=front_directory,
                side_directory=side_directory,
                preview_directory=preview_directory,
                front_position=args.front_position,
                min_aspect_ratio=args.min_aspect_ratio,
                color_tolerance=args.color_tolerance,
                search_start_ratio=args.search_start_ratio,
                search_end_ratio=args.search_end_ratio,
                max_foreground_ratio=args.max_foreground_ratio,
                minimum_separator_width=args.minimum_separator_width,
                trim=not args.no_trim,
                trim_padding=args.trim_padding,
                overwrite=args.overwrite,
            )
            report_rows.append(row)
            success_count += 1

            print(
                f"[成功] {source_path.name}: "
                f"分隔带 x={row['separator_start']}.."
                f"{row['separator_end']}"
            )

        except Exception as error:
            skipped_count += 1
            report_rows.append(
                {
                    "source": source_path.name,
                    "status": "skipped",
                    "source_width": "",
                    "source_height": "",
                    "aspect_ratio": "",
                    "separator_start": "",
                    "separator_end": "",
                    "front_width": "",
                    "front_height": "",
                    "side_width": "",
                    "side_height": "",
                    "background_rgb": "",
                    "message": str(error),
                }
            )
            print(f"[跳过] {source_path.name}: {error}")

    report_path = output_directory / "split_report.csv"
    fieldnames = [
        "source",
        "status",
        "source_width",
        "source_height",
        "aspect_ratio",
        "separator_start",
        "separator_end",
        "front_width",
        "front_height",
        "side_width",
        "side_height",
        "background_rgb",
        "message",
    ]

    with report_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n处理结束")
    print(f"成功：{success_count}")
    print(f"跳过：{skipped_count}")
    print(f"预览目录：{preview_directory}")
    print(f"报告文件：{report_path}")
    print(
        "请先检查 previews 中的红、蓝竖线，"
        "确认分隔带识别正确后再开始标注。"
    )


if __name__ == "__main__":
    main()