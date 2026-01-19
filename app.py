import time
import io
import os
import traceback

import numpy as np
import pandas as pd
import duckdb  # kept for parity (not used directly in these benches)
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

# Optional libs
try:
    import polars as pl
    HAS_POLARS = True
except Exception:
    pl = None
    HAS_POLARS = False

# FireDucks new API: import the pandas shim
try:
    import fireducks.pandas as fdpd
    HAS_FIREDUCKS = True
except Exception:
    fdpd = None
    HAS_FIREDUCKS = False

# -------------------------
# Basic utils / data gen
# -------------------------
def generate_data(n_rows: int, n_groups: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ids = np.arange(n_rows)
    categories = rng.integers(0, n_groups, size=n_rows)
    categories = np.array([f"cat_{c}" for c in categories])
    value1 = rng.normal(0, 1, size=n_rows)
    value2 = rng.normal(10, 5, size=n_rows)
    start_date = np.datetime64("2020-01-01")
    dates = start_date + rng.integers(0, 365, size=n_rows).astype("timedelta64[D]")

    return pd.DataFrame(
        {"id": ids, "category": categories, "value1": value1, "value2": value2, "date": dates}
    )

def time_function(fn, repeats=3):
    repeats = int(max(1, repeats))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)
    return float(np.mean(times)), float(np.std(times)), [float(t) for t in times]

# -------------------------
# FireDucks helpers
# -------------------------
def ensure_fireducks_from_pandas(df: pd.DataFrame):
    """
    Convert a pandas DataFrame into a FireDucks-backed pandas object (shim).
    """
    if not HAS_FIREDUCKS:
        raise RuntimeError("FireDucks (fireducks.pandas) not installed")

    # Try common constructors
    try:
        return fdpd.DataFrame(df)
    except Exception:
        pass

    try:
        if hasattr(fdpd, "from_pandas"):
            return fdpd.from_pandas(df)
    except Exception:
        pass

    raise RuntimeError("Could not construct FireDucks DataFrame from pandas with current shim")

def materialize_fireducks(obj):
    """
    Convert FireDucks result to pandas if possible for fair inspection.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if HAS_FIREDUCKS:
        try:
            if hasattr(obj, "to_pandas"):
                return obj.to_pandas()
        except Exception:
            pass
    return obj

# -------------------------
# Benchmark helpers
# -------------------------
def build_result(op_name, pandas_stats, polars_stats, fireducks_stats):
    p_mean, p_std, p_runs = pandas_stats if pandas_stats else (None, None, None)
    pl_mean, pl_std, pl_runs = polars_stats if polars_stats else (None, None, None)
    fd_mean, fd_std, fd_runs = fireducks_stats if fireducks_stats else (None, None, None)

    speed_pl = (p_mean / pl_mean) if (p_mean and pl_mean and pl_mean > 0) else None
    speed_fd = (p_mean / fd_mean) if (p_mean and fd_mean and fd_mean > 0) else None

    return {
        "operation": op_name,
        "pandas_mean_s": p_mean,
        "pandas_std_s": p_std,
        "pandas_runs": p_runs,
        "polars_mean_s": pl_mean,
        "polars_std_s": pl_std,
        "polars_runs": pl_runs,
        "fireducks_mean_s": fd_mean,
        "fireducks_std_s": fd_std,
        "fireducks_runs": fd_runs,
        "speedup_polars_over_pandas": speed_pl,
        "speedup_fireducks_over_pandas": speed_fd,
    }

# -------------------------
# Bench functions (all kept)
# -------------------------
def bench_filter(df: pd.DataFrame, repeats=3):
    def p_op():
        _ = df[(df["value1"] > 0.5) & (df["category"] == df["category"].iloc[0])]

    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        pl_df = pl.from_pandas(df)
        def pl_op():
            first_cat = pl_df["category"][0]
            _ = pl_df.filter((pl.col("value1") > 0.5) & (pl.col("category") == first_cat)).to_pandas()
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            fd_df = ensure_fireducks_from_pandas(df)
            def fd_op():
                res = fd_df[(fd_df["value1"] > 0.5) & (fd_df["category"] == fd_df["category"].iloc[0])]
                _ = materialize_fireducks(res)
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            fd_stats = None

    return build_result("Filter", p_stats, pl_stats, fd_stats)

def bench_groupby(df: pd.DataFrame, repeats=3):
    def p_op():
        _ = df.groupby("category")[["value1", "value2"]].mean()

    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        pl_df = pl.from_pandas(df)
        def pl_op():
            _ = pl_df.group_by("category").agg([pl.col("value1").mean(), pl.col("value2").mean()]).to_pandas()
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            fd_df = ensure_fireducks_from_pandas(df)
            def fd_op():
                res = fd_df.group_by("category")[["value1", "value2"]].mean()
                _ = materialize_fireducks(res)
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            fd_stats = None

    return build_result("Groupby mean", p_stats, pl_stats, fd_stats)

def bench_join(df: pd.DataFrame, repeats=3):
    categories = df["category"].unique()
    rng = np.random.default_rng(123)
    dim_df = pd.DataFrame({"category": categories, "weight": rng.uniform(0.5, 2.0, len(categories))})

    def p_op():
        _ = df.merge(dim_df, on="category", how="left")

    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        pl_df = pl.from_pandas(df)
        pl_dim = pl.from_pandas(dim_df)
        def pl_op():
            _ = pl_df.join(pl_dim, on="category", how="left").to_pandas()
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            fd_df = ensure_fireducks_from_pandas(df)
            fd_dim = ensure_fireducks_from_pandas(dim_df)
            def fd_op():
                res = fd_df.merge(fd_dim, on="category", how="left")
                _ = materialize_fireducks(res)
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            fd_stats = None

    return build_result("Join on category", p_stats, pl_stats, fd_stats)

def bench_fillna(df: pd.DataFrame, repeats=3):
    def p_op():
        _ = df.fillna(0)
    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        pl_df = pl.from_pandas(df)
        def pl_op():
            _ = pl_df.fill_null(0).to_pandas()
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            fd_df = ensure_fireducks_from_pandas(df)
            def fd_op():
                res = fd_df.fillna(0)
                _ = materialize_fireducks(res)
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            fd_stats = None

    return build_result("Fill NA / fillna", p_stats, pl_stats, fd_stats)

def bench_dropna(df: pd.DataFrame, repeats=3):
    def p_op():
        _ = df.dropna()
    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        pl_df = pl.from_pandas(df)
        def pl_op():
            _ = pl_df.drop_nulls().to_pandas()
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            fd_df = ensure_fireducks_from_pandas(df)
            def fd_op():
                res = fd_df.dropna()
                _ = materialize_fireducks(res)
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            fd_stats = None

    return build_result("Drop NA / dropna", p_stats, pl_stats, fd_stats)

def bench_sort(df: pd.DataFrame, repeats=3):
    def p_op():
        _ = df.sort_values("value1")
    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        pl_df = pl.from_pandas(df)
        def pl_op():
            _ = pl_df.sort("value1").to_pandas()
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            fd_df = ensure_fireducks_from_pandas(df)
            def fd_op():
                res = fd_df.sort_values("value1")
                _ = materialize_fireducks(res)
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            fd_stats = None

    return build_result("Sort by value1", p_stats, pl_stats, fd_stats)

def bench_describe(df: pd.DataFrame, repeats=3):
    def p_op():
        _ = df.describe()
    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        pl_df = pl.from_pandas(df)
        def pl_op():
            _ = pl_df.describe().to_pandas()
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            fd_df = ensure_fireducks_from_pandas(df)
            def fd_op():
                res = fd_df.describe()
                _ = materialize_fireducks(res)
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            fd_stats = None

    return build_result("Describe()", p_stats, pl_stats, fd_stats)

def bench_read_csv(df: pd.DataFrame, repeats=3):
    path = "temp_bench.csv"
    df.to_csv(path, index=False)

    def p_op():
        _ = pd.read_csv(path)
    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        def pl_op():
            _ = pl.read_csv(path).to_pandas()
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            def fd_op():
                res = fdpd.read_csv(path)
                _ = materialize_fireducks(res)
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            try:
                def fd_op_fb():
                    res = fdpd.DataFrame(pd.read_csv(path))
                    _ = materialize_fireducks(res)
                fd_stats = time_function(fd_op_fb, repeats)
            except Exception:
                fd_stats = None

    try:
        os.remove(path)
    except Exception:
        pass

    return build_result("Read CSV", p_stats, pl_stats, fd_stats)

def bench_read_parquet(df: pd.DataFrame, repeats=3):
    path = "temp_bench.parquet"
    df.to_parquet(path, index=False)

    def p_op():
        _ = pd.read_parquet(path)
    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        def pl_op():
            _ = pl.read_parquet(path).to_pandas()
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            def fd_op():
                res = fdpd.read_parquet(path)
                _ = materialize_fireducks(res)
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            try:
                def fd_op_fb():
                    res = fdpd.DataFrame(pd.read_parquet(path))
                    _ = materialize_fireducks(res)
                fd_stats = time_function(fd_op_fb, repeats)
            except Exception:
                fd_stats = None

    try:
        os.remove(path)
    except Exception:
        pass

    return build_result("Read Parquet", p_stats, pl_stats, fd_stats)

def bench_write_parquet(df: pd.DataFrame, repeats=3):
    def p_op():
        df.to_parquet("temp_pd.parquet")
    p_stats = time_function(p_op, repeats)

    pl_stats = None
    if HAS_POLARS:
        pl_df = pl.from_pandas(df)
        def pl_op():
            pl_df.write_parquet("temp_pl.parquet")
        pl_stats = time_function(pl_op, repeats)

    fd_stats = None
    if HAS_FIREDUCKS:
        try:
            fd_df = ensure_fireducks_from_pandas(df)
            def fd_op():
                if hasattr(fd_df, "to_parquet"):
                    fd_df.to_parquet("temp_fd.parquet")
                else:
                    materialize_fireducks(fd_df).to_parquet("temp_fd.parquet")
            fd_stats = time_function(fd_op, repeats)
        except Exception:
            fd_stats = None

    for p in ["temp_pd.parquet", "temp_pl.parquet", "temp_fd.parquet"]:
        try:
            os.remove(p)
        except Exception:
            pass

    return build_result("Write Parquet", p_stats, pl_stats, fd_stats)

# -------------------------
# UI helpers: chart and images
# -------------------------
def generate_chart_three(result):
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = []
    values = []
    if result["pandas_mean_s"] is not None:
        labels.append("Pandas")
        values.append(result["pandas_mean_s"])
    if result["polars_mean_s"] is not None:
        labels.append("Polars")
        values.append(result["polars_mean_s"])
    if result["fireducks_mean_s"] is not None:
        labels.append("FireDucks")
        values.append(result["fireducks_mean_s"])
    ax.bar(labels, values)
    ax.set_ylabel("Time (s)")
    ax.set_title(result["operation"])
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, f"{v:.4f}s", ha='center')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def generate_speedbars(result):
    """
    Horizontal bars showing relative speed. Lower time = longer 'speed' bar.
    We'll normalize with the fastest (smallest) time.
    """
    # Collect engines & times
    engines = []
    times = []
    if result["pandas_mean_s"] is not None:
        engines.append("Pandas"); times.append(result["pandas_mean_s"])
    if result["polars_mean_s"] is not None:
        engines.append("Polars"); times.append(result["polars_mean_s"])
    if result["fireducks_mean_s"] is not None:
        engines.append("FireDucks"); times.append(result["fireducks_mean_s"])

    if len(times) == 0:
        # return a small empty image
        img = Image.new("RGB", (600, 80), color=(240,240,240))
        return img

    fastest = min(times)
    # speed multiplier relative to pandas baseline (if pandas present)
    baseline = result["pandas_mean_s"] if result["pandas_mean_s"] else fastest

    # Normalize lengths: invert times so smaller time -> bigger bar
    inv = [fastest / t for t in times]
    max_inv = max(inv)
    lengths = [int(500 * (v / max_inv)) for v in inv]

    fig, ax = plt.subplots(figsize=(6, len(engines) * 0.6 + 0.5))
    y_pos = np.arange(len(engines))

    ax.barh(y_pos, lengths, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(engines)
    ax.invert_yaxis()  # fastest on top
    ax.set_xlabel("Relative speed (normalized to fastest)")
    # Annotate multiplier and actual time
    for i, (l, t) in enumerate(zip(lengths, times)):
        mult = baseline / t if baseline and t else None
        label = f"{t:.4f}s"
        if mult:
            label += f" ({mult:.2f}x vs baseline)"
        ax.text(l + 6, i, label, va='center')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def format_result_md(result):
    md = f"### 🔬 {result['operation']}\n\n"
    md += "| Engine | Mean (s) | Std (s) |\n|---|---:|---:|\n"
    md += f"| Pandas | `{result['pandas_mean_s']}` | `{result['pandas_std_s']}` |\n"
    md += f"| Polars | `{result['polars_mean_s']}` | `{result['polars_std_s']}` |\n"
    md += f"| FireDucks | `{result['fireducks_mean_s']}` | `{result['fireducks_std_s']}` |\n\n"
    if result["speedup_polars_over_pandas"]:
        md += f"- Polars speedup over Pandas: **{result['speedup_polars_over_pandas']:.2f}x**\n"
    if result["speedup_fireducks_over_pandas"]:
        md += f"- FireDucks speedup over Pandas: **{result['speedup_fireducks_over_pandas']:.2f}x**\n"
    md += "\n<details><summary>Raw runs</summary>\n\n"
    md += f"- Pandas runs: `{result['pandas_runs']}`\n"
    md += f"- Polars runs: `{result['polars_runs']}`\n"
    md += f"- FireDucks runs: `{result['fireducks_runs']}`\n"
    md += "\n</details>\n"
    return md

def fastest_engine_badge(result):
    engines = []
    times = []
    if result["pandas_mean_s"] is not None:
        engines.append("Pandas"); times.append(result["pandas_mean_s"])
    if result["polars_mean_s"] is not None:
        engines.append("Polars"); times.append(result["polars_mean_s"])
    if result["fireducks_mean_s"] is not None:
        engines.append("FireDucks"); times.append(result["fireducks_mean_s"])

    if not engines:
        return "<div style='padding:8px;background:#f8d7da;color:#721c24;border-radius:6px'>No engines available</div>"

    idx = int(np.argmin(times))
    fastest = engines[idx]
    time_val = times[idx]
    html = f"""
    <div style="display:inline-block;padding:10px 14px;border-radius:8px;background:#0f172a;color:#fff">
      <strong>Fastest:</strong> {fastest} — {time_val:.4f}s
    </div>
    """
    return html

# -------------------------
# Dispatcher map
# -------------------------
OPERATION_MAP = {
    "Filter": bench_filter,
    "Groupby": bench_groupby,
    "Join": bench_join,
    "Fillna": bench_fillna,
    "Dropna": bench_dropna,
    "Sort": bench_sort,
    "Describe": bench_describe,
    "Read CSV": bench_read_csv,
    "Read Parquet": bench_read_parquet,
    "Write Parquet": bench_write_parquet,
}

def run_benchmark_dispatch(operation, df, repeats):
    if operation not in OPERATION_MAP:
        raise ValueError("Unsupported operation")
    fn = OPERATION_MAP[operation]
    return fn(df, repeats)

# -------------------------
# Gradio UI (Option A layout)
# -------------------------
theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

with gr.Blocks(title="Pandas vs Polars vs FireDucks Benchmark", theme=theme) as demo:
    gr.Markdown("# 🐼 vs 🔥 vs ⚡ Pandas vs Polars vs FireDucks — Benchmark playground")

    with gr.Tabs():
        with gr.Tab("Synthetic dataset"):
            # Controls
            dataset_size = gr.Radio(["100k", "500k", "2M"], value="100k", label="Dataset size")
            operation = gr.Dropdown(list(OPERATION_MAP.keys()), value="Filter", label="Operation")
            repeats = gr.Slider(1, 7, value=3, label="Repeats")
            run_btn = gr.Button("Run benchmark")

            # OUTPUT LAYOUT (Option A): chart top -> speedbars -> fastest badge -> markdown
            chart_out = gr.Image(label="Timing chart (lower is better)", height=300, width=600)
            speedbars_out = gr.Image(label="Relative speedbars (fastest normalized to 1)", height=300, width=600)
            fastest_out = gr.HTML(label="Fastest engine")
            md_out = gr.Markdown()

            def run_synth(size, op, reps):
                # check optional libs
                missing = []
                if not HAS_POLARS:
                    missing.append("polars")
                if not HAS_FIREDUCKS:
                    missing.append("fireducks (fireducks.pandas shim)")
                if missing:
                    # return friendly warning in place of outputs
                    warn = f"⚠ Missing libraries: {', '.join(missing)}. Add them to requirements.txt if you want those engines tested."
                    # for images, return small placeholder image with warning text
                    img = Image.new("RGB", (800, 200), color=(250,250,250))
                    return img, img, f"<div style='color:#b45309;padding:10px'>{warn}</div>", f"**Warning**: {warn}"

                n = {"100k": 100_000, "500k": 500_000, "2M": 2_000_000}[size]
                df = generate_data(n)
                result = run_benchmark_dispatch(op, df, int(reps))

                # Build visuals
                chart = generate_chart_three(result)
                speedbars = generate_speedbars(result)
                fastest_html = fastest_engine_badge(result)
                md = format_result_md(result)
                return chart, speedbars, fastest_html, md

            run_btn.click(run_synth, [dataset_size, operation, repeats], [chart_out, speedbars_out, fastest_out, md_out])

        with gr.Tab("Custom dataset"):
            file_in = gr.File(label="Upload CSV / Parquet / Feather / Arrow", file_types=['.csv', '.parquet', '.feather', '.arrow'])
            operation_c = gr.Dropdown(list(OPERATION_MAP.keys()), value="Filter", label="Operation")
            repeats_c = gr.Slider(1, 7, value=3, label="Repeats")
            run_btn_c = gr.Button("Run on uploaded dataset")

            chart_out_c = gr.Image(label="Timing chart")
            speedbars_out_c = gr.Image(label="Relative speedbars")
            fastest_out_c = gr.HTML(label="Fastest engine")
            md_out_c = gr.Markdown()

            def run_custom(file, op, reps):
                if file is None:
                    img = Image.new("RGB", (800, 200), color=(250,250,250))
                    return img, img, "<div style='color:#b45309;padding:10px'>Upload a dataset file first.</div>", "Upload a dataset file first."
                fname = file.name
                try:
                    if fname.endswith(".csv"):
                        df = pd.read_csv(fname)
                    elif fname.endswith(".parquet"):
                        df = pd.read_parquet(fname)
                    elif fname.endswith(".feather") or fname.endswith(".arrow"):
                        df = pd.read_feather(fname)
                    else:
                        return Image.new("RGB", (800,200),(250,250,250)), Image.new("RGB",(800,200),(250,250,250)), "<div>Unsupported file format</div>", "Unsupported file format"
                except Exception as e:
                    return Image.new("RGB", (800,200),(250,250,250)), Image.new("RGB",(800,200),(250,250,250)), f"<div>Error reading file: {e}</div>", f"Error reading file: {e}"

                result = run_benchmark_dispatch(op, df, int(reps))
                chart = generate_chart_three(result)
                speedbars = generate_speedbars(result)
                fastest_html = fastest_engine_badge(result)
                md = format_result_md(result)
                return chart, speedbars, fastest_html, md

            run_btn_c.click(run_custom, [file_in, operation_c, repeats_c], [chart_out_c, speedbars_out_c, fastest_out_c, md_out_c])

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get("PORT", 7860)))
