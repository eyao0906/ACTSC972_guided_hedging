ACTSC 972 - Guided deep hedging assignment

Read assignment.pdf for the tasks, experiments, deadlines, and submission
requirements. Keep the supplied files together in the extracted folder.

SETUP
Python 3.11 or 3.12 is recommended. A CPU is sufficient for the guided
experiments. Open a terminal in this folder.

macOS/Linux:
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m jupyter lab

Windows (Command Prompt):
    py -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt
    python -m jupyter lab

Select this environment as the notebook kernel in Jupyter or VS Code.
For later sessions, activate the environment and start Jupyter again.
You may use an existing compatible course environment instead.

RUNNING
Start with pytorch_warmup.ipynb if PyTorch is new to you, then open
starter.ipynb and follow its instructions. Unfinished TODO cells deliberately
raise NotImplementedError until you supply the required code.
Results are saved in outputs/ by default.

EXPORTING THE NOTEBOOK TO PDF
After completing the notebook, restart its kernel, run all cells in order,
and save it with its outputs. Use the editor's PDF export if available.
Alternatively, run this command in the activated environment:
    python -m jupyter nbconvert --to html starter.ipynb
Open starter.html in a browser and choose Print, then Save as PDF.
Check that all code and outputs are visible, readable, and not clipped.

report_template.tex and presentation_outline.txt are optional writing aids.
