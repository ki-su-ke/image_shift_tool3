#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG / JPG 一括処理ツール（Pillow版・一発計算版）
はがき宛名面の文字を、rect1にかぶらないように上へ一発シフトする。
ループを排除した叩き台として作成。
同じ結果がでるかは後で確認していこう。
"""

"""
主な変更点
項目                        旧              新
ループ                      最大100回       なし
走査                        何度も          原則1回（scan_content_bounds）
移動量                      2pxずつ         bottom - rect1.y1 + 1 で一発
rect2/rect3の動的処理       あり            廃止（left/rightを決め打ち）
上限判定                    ループ中に      y1<227top - needed と upper_limit_y で事前判定

使い方の例
Bashpython image_shift_oneshot.py /path/to/images \
  --rect1 100,300,500,450 \
  --upper-limit-y 227 \
  --sender-zip-rect 20,500,200,560 \
  --left-limit-x 0 \
  --y-space 4
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Iterable, Optional
from datetime import datetime

from PIL import Image
from openpyxl import Workbook  # pip install openpyxl
import numpy as np


Rect = Tuple[int, int, int, int]  # left, top, right, bottom (半開区間)


@dataclass
class Config:
    input_dir: Path
    rect1: Rect                     # 監視矩形（ここには文字を残さない）
    upper_limit_y: int              # 郵便番号枠下端（これより上に出てはいけない）
    sender_zip_rect: Rect           # 差出人郵便番号枠（走査から除外）
    left_limit_x: int = 0           # 移動対象の左端（決め打ち）
    right_limit_x: Optional[int] = None  # 移動対象の右端（Noneなら画像右端）
    white_threshold: int = 255      # 255=厳密一致。250〜255で揺らぎ許容
    y_space: int = 0                # 正常終了後の追加上方向移動量
    output_subfolder: str = "OUT"
    overwrite: bool = True
    


# ---------------------------
# ユーティリティ
# ---------------------------
def parse_rect(text: str) -> Rect:
    try:
        left, top, right, bottom = [int(v.strip()) for v in text.split(",")]
    except Exception:
        raise argparse.ArgumentTypeError("rect は left, top, right, bottom 形式で指定してください")
    
    if right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("rect の幅・高さが正になるよう指定してください")
    return left, top, right, bottom


def clamp(v: int, lo: int, hi: int) -> int:
    """
    指定された値 v を lo〜hi の範囲にクリップする。
    """
    return max(lo, min(hi, v))


def clip_rect_to_image(rect: Rect, width: int, height: int) -> Optional[Rect]:
    """
    rect を画像サイズにクリップする。
    クリップ後の矩形が無効（幅または高さが0以下）なら None を返す。
    """
    left, top, right, bottom = rect
    left_c, top_c = clamp(left, 0, width), clamp(top, 0, height)
    right_c, bottom_c = clamp(right, 0, width), clamp(bottom, 0, height)
    if right_c <= left_c or bottom_c <= top_c:
        return None
    return left_c, top_c, right_c, bottom_c


def list_image_files(dirpath: Path, output_subfolder: str) -> Iterable[Path]:
    """
    指定フォルダ内の PNG/JPG ファイルを再帰的に列挙する。
    output_subfolder 内のファイルは除外する。
    """
    for p in dirpath.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        if output_subfolder in p.relative_to(dirpath).parts:
            continue
        
        yield p
        

def scan_content_bounds(
    img: Image.Image,
    scan_rect: Rect,
    exclude_rect: Optional[Rect],
    white_threshold: int,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    scan_rect 内を走査し、exclude_rect を除いた範囲の
    top_nonwhite, bottom_nonwhite, left_nonwhite を返す。<br>
    非白が1つもなければ (None, None, None)。
    """
    rgb = img.convert("RGB")
    left, top, right, bottom = scan_rect
    # ex = exclude_rect # Pythonだとimmutableでもないし変数名を短くするくらいしか意味がないので割愛するよ
    
    top_nonwhite = None
    bottom_nonwhite = None
    left_nonwhite = None
    
    if exclude_rect is not None:
        ex_left, ex_top, ex_right, ex_bottom = exclude_rect
    else:
        ex_left = ex_top = ex_right = ex_bottom = -1  # 絶対にヒットしない値
        
    for y in range(top, bottom):
        for x in range(left, right):
            # if exclude_rect is not None:
            #     ex_left, ex_top, ex_right, ex_bottom = exclude_rect
            #     if ex_left <= x < ex_right and ex_top <= y < ex_bottom:
            #         continue
            # ↑この部分はループの外に出す
            #
            if ex_left <= x < ex_right and ex_top <= y < ex_bottom:
                continue
            
            r, g, b = rgb.getpixel((x, y))
            if r < white_threshold or g < white_threshold or b < white_threshold:
                if top_nonwhite is None or y < top_nonwhite:
                    top_nonwhite = y
                if bottom_nonwhite is None or y > bottom_nonwhite:
                    bottom_nonwhite = y
                if left_nonwhite is None or x < left_nonwhite:
                    left_nonwhite = x
    
    return top_nonwhite, bottom_nonwhite, left_nonwhite


def scan_content_bounds_np(
    img: Image.Image,
    scan_rect: Rect,
    exclude_rect: Optional[Rect],
    white_threshold: int,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    scan_rect 内を走査し、exclude_rect を除いた範囲の
    top_nonwhite, bottom_nonwhite, left_nonwhite を返す。
    非白が1つもなければ (None, None, None)。
    NumPyベクトル化により、300dpiハガキサイズでも高速に動作するかも？
    """
    left, top, right, bottom = scan_rect
    
    # PIL Image → NumPy配列 (H, W, 3)
    rgb = img.convert("RGB")
    arr = np.array(rgb)  # shape: (height, width, 3)
    
    # 走査範囲のスライス
    sub_arr = arr[top:bottom, left:right]  # shape: (height, width, 3)
    
    # 非白マスク: いずれかのチャンネルが閾値未満
    if white_threshold >= 255:
        mask = np.any(sub_arr < 255, axis=2)  # shape: (height, width)
    else:
        mask = np.any(sub_arr < white_threshold, axis=2)

    # 除外矩形のマスク
    if exclude_rect is not None:
        ex_left, ex_top, ex_right, ex_bottom = exclude_rect
        #
        # グローバル座標 → ローカル座標に変換
        ex_top_local = max(0, ex_top - top)
        ex_bottom_local = min(bottom - top, ex_bottom - top)
        ex_left_local = max(0, ex_left - left)
        ex_right_local = min(right - left, ex_right - left)
        
        if ex_bottom_local > ex_top_local and ex_right_local > ex_left_local:
            mask[ex_top_local:ex_bottom_local, ex_left_local:ex_right_local] = False
    
    # 非白ピクセルのインデックスを取得
    nonwhite_indices = np.where(mask)
    
    if len(nonwhite_indices[0]) == 0:
        # 非白が1つもない場合なら Tuple[None, None, None] を返す
        return None, None, None
    
    # ローカル座標 → グローバル座標に変換
    top_nonwhite = int(nonwhite_indices[0].min() + top)
    bottom_nonwhite = int(nonwhite_indices[0].max() + top)
    left_nonwhite = int(nonwhite_indices[1].min() + left)
    
    return top_nonwhite, bottom_nonwhite, left_nonwhite


def shift_rect_up_and_fill_white(
    img: Image.Image,
    rect: Rect,
    shift_pixels: int
) -> Image.Image:
    """
    指定矩形を上方向に shift_pixels だけシフトし、元の位置を白で塗りつぶす。<br>
    画像の上端を超える場合は、超えない範囲でシフトする。
    """
    width, height = img.size
    clipped_rect = clip_rect_to_image(rect, width, height)
    if clipped_rect is None or shift_pixels <= 0:
        return img
    
    left, top, right, bottom = clipped_rect
    dy = min(shift_pixels, top) # 画像上端を超えない
    if dy <= 0:
        return img
    
    work = img.convert("RGBA")
    region = work.crop((left, top, right, bottom))
    work.paste(region, (left, top - dy))
    
    # 元の下側を白塗り
    strip = Image.new("RGBA", (right - left, dy), (255, 255, 255, 255))
    work.paste(strip, (left, bottom - dy, right, bottom))
    
    return work


def process_image_file(
    path: Path,
    cfg: Config,
    out_dir: Path
 ) -> tuple[bool, str, Optional[int], str]:
    """
    1ファイル処理（ループ無し）
    戻り値: (保存したか, 結果("OK"/"NG"/"処理不要"), 移動量, 理由)
    """
    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        print(f"[SKIP] 読み込み失敗: {path.name}: {e}")
        return False, "処理不要", None, f"読み込み失敗: {e}"
    
    width, height = img.size
    rect1 = clip_rect_to_image(cfg.rect1, width, height)
    rect_sender_zip = clip_rect_to_image(cfg.sender_zip_rect, width, height)
    
    if not rect1:
        return False, "処理不要", None, "rect1範囲外"
    
    # 走査範囲：画像全体から差出人郵便番号枠を除外する想定
    # 必要に応じて走査範囲を絞ってもよい
    scan_rect = (0, 0, width, height)
    # top, bottom, left = scan_content_bounds(img, scan_rect, rect_sender_zip, cfg.white_threshold)
    top, bottom, left = scan_content_bounds_np(img, scan_rect, rect_sender_zip, cfg.white_threshold)
    
    # 非白がなければ何もしない
    if top is None or bottom is None:
        print(f"[NO-OP] 非白なし: {path.name}")
        return False, "処理不要", None, ""
    
    # rect1内に非白があるか（簡易チェック）
    # top〜bottom が rect1 と重なっていれば処理対象
    if bottom < rect1[1] or top >= rect1[3]:
        # rect1と完全に重なっていない
        print(f"[NO-OP] rect1に非白なし: {path.name}")
        return False, "処理不要", None, ""
    
    # 必要な移動量(一番下の非白をrect1の上端より上に出す)
    needed = bottom - rect1[1] + 1
    if needed <= 0:
        print(f"[NO-OP] 移動不要: {path.name}")
        return False, "処理不要", None, ""
    
    # 移動後の上端が上限を超えないか
    new_top = top - needed
    if new_top < cfg.upper_limit_y:
        # 可能な最大移動量にキャップするか、NGにするか
        # ここではNGとする(安全側)
        max_possible = top - cfg.upper_limit_y
        if max_possible <= 0:
            print(f"[NG] 上限超過で移動不可: {path.name}")
            return False, "NG", None, "upper-limit"
        #
        # キャップして動かす場合は以下を有効に
        # needed = max_possible
        # new_top = top - needed
        print(f"[NG] 必要な移動量が上限を超える: {path.name} needed={needed}")
        return False, "NG", None, "upper-limit"
    
    # 移動対象矩形を決定
    left_x = cfg.left_limit_x
    right_x = cfg.right_limit_x if cfg.right_limit_x is not None else width
    #
    # 下端は bottom+1（半開）、上端は top
    move_rect = clip_rect_to_image((left_x, top, right_x, bottom + 1), width, height)
    if move_rect is None:
        return False, "処理不要", None, "移動矩形無効"
    #
    # 一発シフト
    current = shift_rect_up_and_fill_white(img, move_rect, needed)
    moved = needed
    
    # y_space 追加
    if cfg.y_space > 0:
        new_top2 = top - moved - cfg.y_space
        if new_top2 >= cfg.upper_limit_y:
            # 追加移動後の矩形
            move_rect2 = (
                move_rect[0],
                move_rect[1] - moved,
                move_rect[2],
                move_rect[3] - moved,
            )
            move_rect2 = clip_rect_to_image(move_rect2, width, height)
            if move_rect2:
                current = shift_rect_up_and_fill_white(current, move_rect2, cfg.y_space)
                moved += cfg.y_space
        else:
            # y_space を足すと上限超過 → NG にする
            # print(f"[WARN] y_spaceを追加すると上限超過のためスキップ: {path.name}")
            return False, "NG", moved, "y-space-clip-failed"
    
    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    base = path.stem + "_sft"
    result_label = "OK"
    out_path = out_dir / f"{base}.png"
    
    try:
        current.save(out_path)
        print(f"[SAVE] {path.name} -> {out_path.name} (OK, moved={moved})")
        return True, result_label, moved, ""
    except Exception as e:
        print(f"[ERROR] 保存失敗: {out_path}: {e}")
        return False, "処理不要", moved, f"保存失敗: {e}" 


def run(cfg: Config) -> None:
    """
    指定フォルダ内の PNG/JPG ファイルを処理して、結果を Excel に出力する。
    """
    if not cfg.input_dir.exists():
        raise SystemExit(f"入力フォルダが見つかりません: {cfg.input_dir}")
    
    files = list(list_image_files(cfg.input_dir, cfg.output_subfolder))
    if not files:
        print("[INFO] 対象画像がありません（png/jpg）。")
        return
    
    base_out_dir = cfg.input_dir / cfg.output_subfolder
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = base_out_dir / f"{ts}.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["filename", "result", "moved_px", "reason"])
    
    processed = saved = ng_count = 0
    for p in files:
        processed += 1
        rel_dir = p.parent.relative_to(cfg.input_dir)
        out_dir = base_out_dir / rel_dir
        rel_name = str(p.relative_to(cfg.input_dir))
        #
        # ここで処理
        saved_flag, result, moved, reason = process_image_file(p, cfg, out_dir)
        if saved_flag:
            saved += 1
        if result == "NG":
            ng_count += 1
        
        ws.append([rel_name, result, moved if moved is not None else "", reason])
    
    wb.save(log_path)
    print(f"[LOG] Excel 出力完了: {log_path}")
    print(f"[DONE] {processed}件中 {saved}件 保存完了")
    print(f"[SUMMARY] NG: {ng_count}/{processed}件")
    

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="はがき宛名面 一発シフトツール")
    ap.add_argument("input_dir", type=Path, help="入力フォルダ")
    ap.add_argument("--rect1", required=True, type=parse_rect, help="監視矩形 x1,y1,x2,y2")
    ap.add_argument("--upper-limit-y", required=True, type=int, help="郵便番号枠下端y（これより上に出ない）")
    ap.add_argument("--sender-zip-rect", required=True, type=parse_rect, help="差出人郵便番号枠")
    ap.add_argument("--left-limit-x", type=int, default=0, help="移動対象左端 [default=0]")
    ap.add_argument("--right-limit-x", type=int, default=None, help="移動対象右端 [default=画像右端]")
    ap.add_argument("--white-threshold", type=int, default=255, help="白閾値")
    ap.add_argument("--y-space", type=int, default=0, help="正常終了後の追加移動量")
    ap.add_argument("--output-subfolder", type=str, default="OUT", help="出力サブフォルダ")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    
    cfg = Config(
        input_dir=args.input_dir.resolve(),
        rect1=args.rect1,
        upper_limit_y=args.upper_limit_y,
        sender_zip_rect=args.sender_zip_rect,
        left_limit_x=args.left_limit_x,
        right_limit_x=args.right_limit_x,
        white_threshold=args.white_threshold,
        y_space=args.y_space,
        output_subfolder=args.output_subfolder,
    )
    run(cfg)


if __name__ == "__main__":
    main()
