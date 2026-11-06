"""
Comet Python DataManager (`python/comet/dmgr.py`)

Provides read/write support for the Comet Memory-Mapped Binary Tensor (`.bin`) format
and `.csv.gz` matrices, with automatic universe alignment (`dates` x `instruments`)
and full `pandas.DataFrame` compatibility (plus a lightweight fallback when `pandas`
is not installed in the current environment).
"""

from __future__ import annotations
import csv
import gzip
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd  # type: ignore

    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False

CMT_MAGIC = b"CMT1"
CMT_VERSION = 1
CMT_DTYPE_F32 = 1
CMT_DTYPE_F64 = 2
CMT_HEADER_SIZE = 64
CMT_HEADER_FMT = "<4sIIIQQQQQQ"  # 4 + 4*3 + 8*6 = 64 bytes


def hash_coords(items: Sequence[str]) -> int:
    """Deterministic 64-bit FNV-1a hash matching `runtime::dmgr::hash_coords` in Rust."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    for item in items:
        for b in str(item).encode("utf-8"):
            h ^= b
            h = (h * prime) & mask
        h ^= 0xFF
        h = (h * prime) & mask
    return h


class AlignedFrame:
    """
    Lightweight DataFrame-compatible wrapper used when `pandas` is not installed.
    Supports `.values`, `.index`, `.columns`, `.shape`, `.reindex()`, `.to_numpy()`,
    and `.to_pandas()` (when pandas is available).
    """

    def __init__(
        self,
        data: np.ndarray,
        index: Optional[Sequence[str]] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.values = np.asarray(data)
        if self.values.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {self.values.shape}")
        r, c = self.values.shape
        self.index: List[str] = (
            [str(x) for x in index] if index is not None else [str(i) for i in range(r)]
        )
        self.columns: List[str] = (
            [str(x) for x in columns] if columns is not None else [str(j) for j in range(c)]
        )
        if len(self.index) != r or len(self.columns) != c:
            raise ValueError(
                f"Index/columns length ({len(self.index)}, {len(self.columns)}) "
                f"does not match array shape ({r}, {c})"
            )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.values.shape  # type: ignore

    def to_numpy(self) -> np.ndarray:
        return self.values

    def reindex(
        self,
        index: Optional[Sequence[str]] = None,
        columns: Optional[Sequence[str]] = None,
        fill_value: float = np.nan,
    ) -> "AlignedFrame":
        target_dates = [str(x) for x in index] if index is not None else self.index
        target_insts = [str(x) for x in columns] if columns is not None else self.columns
        aligned = align_2d_array(
            self.values,
            self.index,
            self.columns,
            target_dates,
            target_insts,
            fill_value=fill_value,
        )
        return AlignedFrame(aligned, index=target_dates, columns=target_insts)

    def __repr__(self) -> str:
        return (
            f"AlignedFrame(shape={self.shape}, "
            f"index=[{self.index[0] if self.index else ''}..{self.index[-1] if self.index else ''}], "
            f"columns=[{self.columns[0] if self.columns else ''}..{self.columns[-1] if self.columns else ''}])"
        )


def align_2d_array(
    src_data: np.ndarray,
    src_dates: Sequence[str],
    src_insts: Sequence[str],
    dst_dates: Sequence[str],
    dst_insts: Sequence[str],
    fill_value: float = np.nan,
    ffill: bool = False,
) -> np.ndarray:
    """
    Fast vectorized 2D reindexing of `src_data` from `(src_dates, src_insts)`
    onto `(dst_dates, dst_insts)`.
    """
    if (
        len(src_dates) == len(dst_dates)
        and len(src_insts) == len(dst_insts)
        and list(src_dates) == list(dst_dates)
        and list(src_insts) == list(dst_insts)
    ):
        return src_data

    dst_r = len(dst_dates)
    dst_c = len(dst_insts)
    out = np.full((dst_r, dst_c), fill_value, dtype=src_data.dtype)

    date_map = {str(d): i for i, d in enumerate(src_dates)}
    inst_map = {str(ins): j for j, ins in enumerate(src_insts)}

    dst_row_indices = [ r for r, d in enumerate(dst_dates) if str(d) in date_map ]
    src_row_indices = [ date_map[str(dst_dates[r])] for r in dst_row_indices ]

    dst_col_indices = [ c for c, ins in enumerate(dst_insts) if str(ins) in inst_map ]
    src_col_indices = [ inst_map[str(dst_insts[c])] for c in dst_col_indices ]

    if dst_row_indices and dst_col_indices:
        sub = src_data[np.ix_(src_row_indices, src_col_indices)]
        out[np.ix_(dst_row_indices, dst_col_indices)] = sub

    if ffill and dst_r > 1:
        for c in range(dst_c):
            col = out[:, c]
            mask = np.isnan(col)
            if not np.all(mask) and np.any(mask):
                idx = np.where(~mask, np.arange(dst_r), 0)
                np.maximum.accumulate(idx, out=idx)
                valid_start = int(np.argmax(~mask))
                col[valid_start:] = col[idx[valid_start:]]

    return out


class DataManager:
    """
    Python DataManager for reading, writing, and aligning `.cmt` (memory-mapped tensor)
    and `.csv.gz` files with full `pandas.DataFrame` compatibility.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        wrds_dir: Union[str, Path] = "wrds",
        dates: Optional[Sequence[str]] = None,
        instruments: Optional[Sequence[str]] = None,
        use_mmap: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.wrds_dir = Path(wrds_dir)
        self.use_mmap = use_mmap
        self.dates: List[str] = []
        self.instruments: List[str] = []
        self.dates_hash: int = 0
        self.insts_hash: int = 0
        self._discover_or_set_universe(dates, instruments)

    def _discover_or_set_universe(
        self,
        dates: Optional[Sequence[str]],
        instruments: Optional[Sequence[str]],
    ) -> None:
        d_list = [str(x).strip() for x in dates] if dates is not None else None
        i_list = [str(x).strip() for x in instruments] if instruments is not None else None

        # 1. Check info.json / universe.json in data_dir
        for fname in ("universe.json", "info.json"):
            fpath = self.data_dir / fname
            if fpath.exists():
                try:
                    info = json.loads(fpath.read_text(encoding="utf-8"))
                    if d_list is None and "dates" in info:
                        d_list = [str(x).strip() for x in info["dates"]]
                    elif d_list is None and "dates_file" in info:
                        d_list = self._read_lines(Path(info["dates_file"]))
                    if i_list is None and "instruments" in info:
                        i_list = [str(x).strip() for x in info["instruments"]]
                    elif i_list is None and "instruments_file" in info:
                        i_list = self._read_lines(Path(info["instruments_file"]))
                except Exception:
                    pass

        # 2. Check universe_dates.csv / universe_instruments.csv
        for base in (self.data_dir, self.wrds_dir, self.data_dir.parent / "wrds"):
            if d_list is None and (base / "universe_dates.csv").exists():
                d_list = self._read_lines(base / "universe_dates.csv")
            if i_list is None and (base / "universe_instruments.csv").exists():
                i_list = self._read_lines(base / "universe_instruments.csv")

        self.dates = d_list or []
        self.instruments = i_list or []
        self.dates_hash = hash_coords(self.dates) if self.dates else 0
        self.insts_hash = hash_coords(self.instruments) if self.instruments else 0

    @staticmethod
    def _read_lines(path: Path) -> List[str]:
        if not path.exists():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def set_universe(self, dates: Sequence[str], instruments: Sequence[str]) -> None:
        """Updates the target universe used for alignment."""
        self.dates = [str(x).strip() for x in dates]
        self.instruments = [str(x).strip() for x in instruments]
        self.dates_hash = hash_coords(self.dates)
        self.insts_hash = hash_coords(self.instruments)

    # -------------------------------------------------------------------------
    # Binary (.bin) Read / Write
    # -------------------------------------------------------------------------
    def write_bin(
        self,
        name_or_path: Union[str, Path],
        data: Any,
        dates: Optional[Sequence[str]] = None,
        instruments: Optional[Sequence[str]] = None,
        dtype: str = "f64",
        align_to_universe: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Writes a `pandas.DataFrame`, `AlignedFrame`, or 2D `numpy.ndarray` to a `.bin` file.
        - If `data` is a `pandas.DataFrame` or `AlignedFrame`, `dates` and `instruments` are
          automatically extracted from `data.index` and `data.columns` unless overridden.
        - If `align_to_universe=True`, reindexes `data` onto `(self.dates, self.instruments)`
          before writing so the resulting `.bin` file takes the $O(1)$ zero-copy fast path in Rust.
        """
        out_path = self._resolve_bin_path(name_or_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract array, dates, and instruments from DataFrame / AlignedFrame / ndarray
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            arr = data.to_numpy()
            src_dates = [str(x) for x in (dates if dates is not None else data.index.tolist())]
            src_insts = [str(x) for x in (instruments if instruments is not None else data.columns.tolist())]
        elif isinstance(data, AlignedFrame):
            arr = data.values
            src_dates = [str(x) for x in (dates if dates is not None else data.index)]
            src_insts = [str(x) for x in (instruments if instruments is not None else data.columns)]
        else:
            arr = np.asarray(data)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")
            src_dates = [str(x) for x in (dates if dates is not None else self.dates)]
            src_insts = [str(x) for x in (instruments if instruments is not None else self.instruments)]

        if align_to_universe:
            if not self.dates or not self.instruments:
                raise ValueError("Cannot align_to_universe when target universe is empty")
            arr = align_2d_array(arr, src_dates, src_insts, self.dates, self.instruments)
            src_dates = list(self.dates)
            src_insts = list(self.instruments)

        rows, cols = arr.shape
        if len(src_dates) != rows or len(src_insts) != cols:
            raise ValueError(
                f"Coordinates length ({len(src_dates)}, {len(src_insts)}) != array shape ({rows}, {cols})"
            )

        if dtype in ("f64", "float64", np.float64):
            dtype_code = CMT_DTYPE_F64
            np_dtype = "<f8"
        elif dtype in ("f32", "float32", np.float32):
            dtype_code = CMT_DTYPE_F32
            np_dtype = "<f4"
        else:
            raise ValueError(f"Unsupported dtype: {dtype} (expected 'f64' or 'f32')")

        meta_obj: Dict[str, Any] = {
            "dates": src_dates,
            "instruments": src_insts,
        }
        if extra_metadata:
            meta_obj.update(extra_metadata)

        meta_bytes = json.dumps(meta_obj, separators=(",", ":")).encode("utf-8")
        meta_len = len(meta_bytes)
        raw_end = CMT_HEADER_SIZE + meta_len
        data_offset = (raw_end + 63) & ~63  # 64-byte alignment
        pad_len = data_offset - raw_end

        d_hash = hash_coords(src_dates)
        i_hash = hash_coords(src_insts)

        header = struct.pack(
            CMT_HEADER_FMT,
            CMT_MAGIC,
            CMT_VERSION,
            dtype_code,
            1,  # flags: bit 0 = has_embedded_coords
            rows,
            cols,
            data_offset,
            meta_len,
            d_hash,
            i_hash,
        )

        arr_contig = np.ascontiguousarray(arr, dtype=np_dtype)
        with open(out_path, "wb") as f:
            f.write(header)
            f.write(meta_bytes)
            if pad_len > 0:
                f.write(b"\x00" * pad_len)
            f.write(memoryview(arr_contig))

        return out_path

    # Alias for backward compatibility
    write_cmt = write_bin

    def read_bin(
        self,
        name_or_path: Union[str, Path],
        align: bool = True,
        ffill: bool = False,
        use_mmap: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Any]]:
        """
        Reads a `.bin` file via `numpy.memmap` (or in-memory buffer) and optionally
        aligns it to `(self.dates, self.instruments)`.
        Returns `(array, dates, instruments, metadata)`.
        """
        path = self._resolve_bin_path(name_or_path)
        mmap_mode = self.use_mmap if use_mmap is None else use_mmap

        with open(path, "rb") as f:
            hdr_bytes = f.read(CMT_HEADER_SIZE)
            if len(hdr_bytes) < CMT_HEADER_SIZE:
                raise ValueError(f"File {path} is too short for CMT1 header")
            (
                magic,
                version,
                dtype_code,
                flags,
                rows,
                cols,
                data_offset,
                meta_len,
                d_hash,
                i_hash,
            ) = struct.unpack(CMT_HEADER_FMT, hdr_bytes)

            if magic != CMT_MAGIC:
                raise ValueError(f"Invalid magic in {path}: {magic!r}")
            if version != CMT_VERSION:
                raise ValueError(f"Unsupported CMT version in {path}: {version}")

            meta_bytes = f.read(meta_len) if meta_len > 0 else b"{}"
            meta = json.loads(meta_bytes.decode("utf-8"))

        np_dtype = np.float64 if dtype_code == CMT_DTYPE_F64 else np.float32
        if mmap_mode:
            arr: np.ndarray = np.memmap(
                path,
                dtype=np_dtype,
                mode="r",
                offset=data_offset,
                shape=(rows, cols),
            )
        else:
            with open(path, "rb") as f:
                f.seek(data_offset)
                raw = f.read(rows * cols * np.dtype(np_dtype).itemsize)
                arr = np.frombuffer(raw, dtype=np_dtype).reshape((rows, cols)).copy()

        src_dates = meta.get("dates", self.dates if rows == len(self.dates) else [str(i) for i in range(rows)])
        src_insts = meta.get("instruments", self.instruments if cols == len(self.instruments) else [str(j) for j in range(cols)])

        if align and self.dates and self.instruments:
            if (
                rows == len(self.dates)
                and cols == len(self.instruments)
                and d_hash == self.dates_hash
                and i_hash == self.insts_hash
                and not ffill
            ):
                return arr, list(self.dates), list(self.instruments), meta

            aligned = align_2d_array(
                np.asarray(arr),
                src_dates,
                src_insts,
                self.dates,
                self.instruments,
                ffill=ffill,
            )
            return aligned, list(self.dates), list(self.instruments), meta

        return arr, list(src_dates), list(src_insts), meta

    # Alias for backward compatibility
    read_cmt = read_bin

    # -------------------------------------------------------------------------
    # High-Level Unified Readers (Pandas / NumPy)
    # -------------------------------------------------------------------------
    def get_numpy(
        self,
        name: str,
        align: bool = True,
        ffill: bool = False,
        use_mmap: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Loads field `name` as a 2D `numpy.ndarray` (or `numpy.memmap` when aligned `.bin` exists).
        Automatically prefers `<data_dir>/<name>.bin` and falls back to `<data_dir>/<name>.csv.gz`.
        """
        bin_path = self.data_dir / f"{name}.bin"
        if bin_path.exists():
            arr, _, _, _ = self.read_bin(bin_path, align=align, ffill=ffill, use_mmap=use_mmap)
            return arr

        csv_gz_path = self.data_dir / f"{name}.csv.gz"
        if csv_gz_path.exists():
            arr, src_d, src_i = self._read_csv_gz(csv_gz_path)
            if align and self.dates and self.instruments:
                arr = align_2d_array(arr, src_d, src_i, self.dates, self.instruments, ffill=ffill)
            return arr

        raise FileNotFoundError(f"Field '{name}' not found at {bin_path} or {csv_gz_path}")

    def get_dataframe(
        self,
        name: str,
        align: bool = True,
        ffill: bool = False,
        use_mmap: Optional[bool] = None,
    ) -> Any:
        """
        Loads field `name` as a `pandas.DataFrame` indexed by `dates` and columned by `instruments`
        (or an `AlignedFrame` if `pandas` is not installed in the Python environment).
        """
        bin_path = self.data_dir / f"{name}.bin"
        if bin_path.exists():
            arr, d_idx, i_cols, _ = self.read_bin(bin_path, align=align, ffill=ffill, use_mmap=use_mmap)
        else:
            csv_gz_path = self.data_dir / f"{name}.csv.gz"
            if not csv_gz_path.exists():
                raise FileNotFoundError(f"Field '{name}' not found at {bin_path} or {csv_gz_path}")
            arr, d_idx, i_cols = self._read_csv_gz(csv_gz_path)
            if align and self.dates and self.instruments:
                arr = align_2d_array(arr, d_idx, i_cols, self.dates, self.instruments, ffill=ffill)
                d_idx = list(self.dates)
                i_cols = list(self.instruments)

        if HAS_PANDAS:
            return pd.DataFrame(arr, index=d_idx, columns=i_cols, copy=False)
        return AlignedFrame(arr, index=d_idx, columns=i_cols)

    def _read_csv_gz(self, path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
        if HAS_PANDAS:
            # Peek first line to check if headerless matrix or labeled DataFrame CSV
            with gzip.open(path, "rt", encoding="utf-8") as f:
                first_row = next(csv.reader(f), [])
            has_labels = False
            if first_row:
                try:
                    if first_row[0].strip() != "":
                        float(first_row[0])
                    if len(first_row) > 1 and first_row[1].strip() != "":
                        float(first_row[1])
                except ValueError:
                    has_labels = True

            if has_labels:
                df = pd.read_csv(path, index_col=0)
                return (
                    df.to_numpy(dtype=np.float64),
                    [str(x) for x in df.index],
                    [str(x) for x in df.columns],
                )
            else:
                df = pd.read_csv(path, header=None)
                arr = df.to_numpy(dtype=np.float64)
                r, c = arr.shape
                d_idx = self.dates if r == len(self.dates) else [str(i) for i in range(r)]
                i_cols = self.instruments if c == len(self.instruments) else [str(j) for j in range(c)]
                return arr, list(d_idx), list(i_cols)

        # Fallback without pandas
        with gzip.open(path, "rt", encoding="utf-8") as f:
            all_rows = list(csv.reader(f))
        if not all_rows:
            return np.empty((0, 0), dtype=np.float64), [], []

        has_labels = False
        try:
            if all_rows[0][0].strip() != "":
                float(all_rows[0][0])
            if len(all_rows[0]) > 1 and all_rows[0][1].strip() != "":
                float(all_rows[0][1])
        except ValueError:
            has_labels = True

        if has_labels:
            i_cols = [c.strip() for c in all_rows[0][1:]]
            d_idx = []
            num_rows = []
            for row in all_rows[1:]:
                if not row:
                    continue
                d_idx.append(row[0].strip())
                num_rows.append([float(x) if x.strip() != "" else np.nan for x in row[1:]])
            return np.array(num_rows, dtype=np.float64), d_idx, i_cols

        num_rows = [[float(x) if x != "" else np.nan for x in row] for row in all_rows]
        arr = np.array(num_rows, dtype=np.float64)
        r, c = arr.shape
        d_idx = self.dates if r == len(self.dates) else [str(i) for i in range(r)]
        i_cols = self.instruments if c == len(self.instruments) else [str(j) for j in range(c)]
        return arr, list(d_idx), list(i_cols)

    def convert_csv_gz_to_bin(self, name: str, dtype: str = "f64") -> Path:
        """Converts `<data_dir>/<name>.csv.gz` into `<data_dir>/<name>.bin`."""
        csv_path = self.data_dir / f"{name}.csv.gz"
        arr, d_idx, i_cols = self._read_csv_gz(csv_path)
        return self.write_bin(
            name,
            arr,
            dates=d_idx,
            instruments=i_cols,
            dtype=dtype,
            align_to_universe=bool(self.dates and self.instruments),
        )

    convert_csv_gz_to_cmt = convert_csv_gz_to_bin

    def convert_all_csv_gz_to_bin(self, dtype: str = "f64") -> List[Path]:
        """Batch converts all `*.csv.gz` files in `self.data_dir` to `.bin`."""
        converted: List[Path] = []
        for p in sorted(self.data_dir.glob("*.csv.gz")):
            stem = p.name[: -len(".csv.gz")]
            out = self.convert_csv_gz_to_bin(stem, dtype=dtype)
            converted.append(out)
        return converted

    convert_all_csv_gz_to_cmt = convert_all_csv_gz_to_bin

    def _resolve_bin_path(self, name_or_path: Union[str, Path]) -> Path:
        p = Path(name_or_path)
        if p.suffix == ".bin" or "/" in str(name_or_path) or "\\" in str(name_or_path):
            return p if p.suffix == ".bin" else p.with_suffix(".bin")
        return self.data_dir / f"{name_or_path}.bin"
