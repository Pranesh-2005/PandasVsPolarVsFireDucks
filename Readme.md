# 🦆 PandasVsPolarVsFireDucks

Welcome to **PandasVsPolarVsFireDucks**!  
This project benchmarks and compares the performance of popular Python data processing libraries: **Pandas**, **Polars**, and (via DuckDB integration) **FireDucks**. The interactive app lets you visualize results and explore performance trade-offs in real time.

---

## 🚀 Introduction

Data processing in Python is powered by several high-performance libraries. **PandasVsPolarVsFireDucks** provides a hands-on, interactive environment to analyze, compare, and visualize speed, memory usage, and more for:
- [Pandas](https://pandas.pydata.org/)
- [Polars](https://pola.rs/)
- [DuckDB](https://duckdb.org/) (for parity/comparison, via FireDucks concept)

The project includes a Gradio-powered web app (`app.py`) to run benchmarks and visualize results.

---

## ✨ Features

- **Benchmarking**: Compare processing speed and efficiency across Pandas, Polars, and DuckDB.
- **Interactive UI**: Gradio web interface for easy exploration of results.
- **Visualization**: Generates plots and images to illustrate performance differences.
- **Extensible**: Easily add new benchmarks or data processing tasks.
- **Optional Support**: Automatically detects and supports Polars if installed.

---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- pip

### Clone the repository

```bash
git clone https://github.com/Pranesh-2005/PandasVsPolarVsFireDucks.git
cd PandasVsPolarVsFireDucks
```

### Install dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `pandas`
- `numpy`
- `duckdb`
- `gradio`
- `matplotlib`
- `Pillow`
- *(Optional)* `polars`

---

## 📈 Usage

### 1. Launch the App

```bash
python app.py
```

### 2. Interact

- Visit the local Gradio web page (URL will be shown on the terminal).
- Select benchmarks to run.
- Visualize and compare results.

### 3. Example

```python
# app.py example snippet
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
```

---

## 🤝 Contributing

We welcome contributions!  
To contribute:

1. **Fork** the repository
2. **Clone** your fork
3. **Create a branch** (`git checkout -b feature/fooBar`)
4. **Commit** your changes
5. **Open a Pull Request**

Please follow [Conventional Commits](https://www.conventionalcommits.org/) and strive for clear, documented code.

---

## 📄 License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for more information.

---

**Happy benchmarking! 🚀**

## License
This project is licensed under the **MIT** License.

---
🔗 GitHub Repo: https://github.com/Pranesh-2005/PandasVsPolarVsFireDucks