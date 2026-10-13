# Predicting Playback Degradation Events

## Run on Windows 11 using VS Code

1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and select **Add Python to PATH**.
2. Install Visual Studio Code from https://code.visualstudio.com/ and install the Microsoft **Python** extension.
3. Extract this ZIP to a folder such as `C:\Users\YourName\Documents\playback_degradation_project`.
4. In VS Code, choose **File > Open Folder** and open that extracted folder.
5. Open **Terminal > New Terminal**.
6. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
7. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
8. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
9. Start the app:
   ```powershell
   streamlit run app.py
   ```
10. A browser tab should open. If not, copy the `Local URL` shown in the terminal (usually http://localhost:8501).

## What it includes

- Dataset overview and target distribution
- Random Forest baseline classifier
- Holdout evaluation: accuracy, precision, recall, F1, ROC-AUC and confusion matrix
- Form to estimate risk for a hypothetical streaming session
- Dataset preview, types, and missing-value checks

## Dataset target

`Playback_Degradation_Event`: 1 = playback degradation event, 0 = smooth playback.

`Session_ID` is excluded from model inputs. The dataset has 50,000 rows and 0 missing cells at initial inspection. The target is imbalanced, so don't rely on accuracy alone.

## Troubleshooting

- `py is not recognized`: reinstall Python and enable **Add Python to PATH**, then restart VS Code.
- `streamlit is not recognized`: activate `.venv` and run `python -m pip install -r requirements.txt`; then use `python -m streamlit run app.py`.
- Missing CSV error: ensure both CSV files are inside the `data` folder.
- Port in use: run `streamlit run app.py --server.port 8502`.
