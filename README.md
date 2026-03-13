# Brachistochrone Problem

This repository is a reproducible **Jupyter Notebook** project for exploring the brachistochrone (curve of fastest descent):
- derive and numerically validate the classical cycloid trajectory,
- compare classical and relativistic speed models,
- export a shareable static HTML report.

## Project files

- `brachistochrone.ipynb`: main analysis notebook (derivations, numerical experiments, plots, tables).
- `export_html.sh`: one-click notebook-to-HTML export script with MathJax compatibility patching.
- `brachistochrone.html`: exported static report page.

---

## 1) Requirements

Recommended environment:
- Python 3.10+
- Linux / macOS / WSL (Windows users can run equivalent commands in PowerShell/Git Bash)

Main dependencies:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `jupyter` / `nbconvert`

---

## 2) Local setup and run (recommended)

### 2.1 Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy notebook nbconvert
```

### 2.2 Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `brachistochrone.ipynb` and run all cells.

### 2.3 Export HTML using the built-in script

> `export_html.sh` calls `.venv/bin/jupyter` by default, so create `.venv` first.

```bash
chmod +x export_html.sh
./export_html.sh
```

Optional explicit input/output:

```bash
./export_html.sh brachistochrone.ipynb brachistochrone.html
```

---

## 3) Why use `export_html.sh`

Compared with calling `jupyter nbconvert` directly, the script also:
1. exports with embedded images (`--embed-images`) for easier sharing,
2. patches MathJax setup to improve formula rendering reliability across browsers.

---

## 4) Deployment options (static hosting)

`brachistochrone.html` is a static file and can be hosted anywhere.

### Option A: GitHub Pages

1. Push this repo to GitHub.
2. Enable **Pages** in repository settings.
3. Choose a publish source (e.g., branch root or `docs/`).
4. Make sure `brachistochrone.html` is served (or rename it to `index.html`).

### Option B: Nginx

```bash
sudo mkdir -p /var/www/brachistochrone
sudo cp brachistochrone.html /var/www/brachistochrone/index.html
```

Minimal server block:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/brachistochrone;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

Apply config:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## 5) CI recommendation

For team workflows (e.g., GitHub Actions):
1. execute the notebook and export HTML on every main branch update,
2. publish the generated report automatically (Pages or object storage).

Best practices:
- pin dependency versions,
- fail-fast on notebook/export errors,
- verify HTML artifact existence before deployment.

---

## 6) Troubleshooting

### `./export_html.sh` says `.venv/bin/jupyter not found`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install notebook nbconvert numpy pandas matplotlib scipy
```

### Formulas render incorrectly
Use `./export_html.sh` instead of raw `nbconvert` to include MathJax patching.

### Chinese text in plots appears garbled
Set a CJK-compatible Matplotlib font in the notebook and install that font on the runtime machine.

---

## 7) Suggested future layout (optional)

```text
.
├── notebooks/
│   └── brachistochrone.ipynb
├── reports/
│   └── brachistochrone.html
├── scripts/
│   └── export_html.sh
├── requirements.txt
└── README.md
```

---

## 8) Quick command checklist

```bash
# 1) bootstrap
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install numpy pandas matplotlib scipy notebook nbconvert

# 2) run notebook
jupyter notebook brachistochrone.ipynb

# 3) export report
chmod +x export_html.sh && ./export_html.sh
```
