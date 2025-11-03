1. Create environment

```bash
python -m venv venv
```

2. Activate Environment

```bash
source venv/bin/activate # linux
.\venv\Scripts\Activate.ps1 # windows
```

3. Install Packages

```bash
pip install -r requirements.txt
```

4. Run the App

```bash
streamlit run src/app.py
```


## Lint & Format

To format the code:

```bash
ruff format .
```

To check for errors and style issues:

```bash
ruff check .
```

To automatically fix issues:

```bash
ruff check . --fix
```