# Capstone – Laptop Recovery & Environment Setup

This file has two parts:

- **Part 1** is for me (Kabir) to do by hand: installers, sign-ins and checks that need clicks.
- **Part 2** is the handoff for Claude Code: full context of what happened and what it needs to finish.

When Part 1 is done, open Claude Code in VS Code inside the Capstone folder and send:

> Read `SETUP_RECOVERY.md` and complete Part 2. Ask me before anything destructive or before pushing.

---

## Part 1 – Manual checklist (me)

Tick each item as you go.

### 1. Safety first
- [ ] Run a **Full scan** with Windows Security (Virus & threat protection → Scan options → Full scan), including the D drive. The reset was caused by malware from pirated games, and restored files may still be infected.
- [ ] Do **not** run old `.exe` installers from the restored data (e.g. the old `Anaconda3-2025.06-0-Windows-x86_64` on the D Desktop). Download fresh ones.

### 2. Git & GitHub
- [ ] Git is installed and `C:\Users\user\.gitconfig` exists. Check identity in a terminal:
  ```
  git config --global user.name
  git config --global user.email
  ```
  If either is empty:
  ```
  git config --global user.name "Kabir Oberoi"
  git config --global user.email "<the email on my GitHub account>"
  ```
- [ ] In VS Code, click the **Accounts** icon (bottom-left) → **Sign in with GitHub**. (Pushing from the terminal already works through the browser login.)

### 3. VS Code
- [x] VS Code installed.
- [x] Old extensions copied from `D:\C drive data\Kabir Oberoi\.vscode\extensions\` to `C:\Users\user\.vscode\extensions\`.
- [x] Capstone folder trusted (Restricted Mode off).
- [ ] Reload VS Code once all extensions finish installing/updating (`Ctrl+Shift+P` → **Reload Window**).
- [ ] Old user settings were **not** restored (no `AppData\Roaming` on D). Try **Accounts → Backup and Sync Settings** to recover them. If there's no backup, set preferences again via `Ctrl+,`, then **leave Settings Sync on**.
- [ ] Decline the Oracle SQL Developer telemetry prompt.
- [ ] Cancel VS Code's "Install uv and Python" prompt. Python will come from conda (step 5).

### 4. Claude Code chat history
- [x] Copied `D:\C drive data\Kabir Oberoi\.claude` to `C:\Users\user\.claude` (deleted `.credentials`).
- [x] Renamed `C:\Users\user\.claude\projects\c--Users-Kabir-Oberoi-Desktop-Capstone` to `d--C-drive-data-Kabir-Oberoi-Desktop-Capstone`.
- [ ] Open `C:\Users\user\.claude\settings.json` in Notepad. Change any `C:\Users\Kabir Oberoi\...` paths to `C:\Users\user\...`.
- [ ] In VS Code, open the folder `D:\C drive data\Kabir Oberoi\Desktop\Capstone` (always open it from this exact path).
- [ ] Click the Claude Code spark icon → **sign in**.
- [ ] Open **past conversations** and check that the old Capstone chats are listed.
  - If they're missing: start one new chat, look in `C:\Users\user\.claude\projects\` for the folder it just created, and move the old `.jsonl` files into that folder.

### 5. Conda (Python)
- [ ] Download a **fresh** Miniconda (or Anaconda) Windows 64-bit installer from anaconda.com and install with default options.
- [ ] Open **Anaconda Prompt** from the Start menu and run:
  ```
  conda init powershell
  ```
- [ ] Close and reopen VS Code. In its terminal, `conda --version` should work.

### 6. NVIDIA GPU driver
- [ ] Install the latest NVIDIA driver for the **RTX 3050 Laptop GPU** (nvidia.com or the NVIDIA app). Fresh Windows installs often only have a basic driver.
- [ ] Restart the laptop, then check in a terminal:
  ```
  nvidia-smi
  ```
  It should show the RTX 3050 and a driver version.

### 7. Hand off to Claude Code
- [ ] Send Claude Code the message at the top of this file.

---

## Part 2 – Handoff for Claude Code

### What happened

- This laptop got an **INACCESSIBLE_BOOT_DEVICE** error (caused by malware from pirated games) and Windows was **reset**.
- A service center restored the old user profile to the D drive at `D:\C drive data\Kabir Oberoi\`. Before the reset everything was on a single C drive under `C:\Users\Kabir Oberoi\`.
- The new Windows account is named **`user`** (`C:\Users\user\`), not `Kabir Oberoi`. Do not rename the Windows profile.
- The project now lives at **`D:\C drive data\Kabir Oberoi\Desktop\Capstone`** and stays there. Old absolute paths of the form `C:\Users\Kabir Oberoi\Desktop\Capstone\...` in code, notebooks or configs are now wrong.
- `AppData\Roaming` was **not** restored, so old VS Code user settings are gone. Only `AppData\Local` came back.
- The old conda installation and the **`hf` conda environment were not restored**. Only an old Anaconda installer exists on the D Desktop; do not use it.
- Claude Code's `.claude` folder was copied to `C:\Users\user\.claude`, and this project's history folder was renamed to `d--C-drive-data-Kabir-Oberoi-Desktop-Capstone`.

### Current state

- Git is installed. Global `.gitconfig` exists at `C:\Users\user\.gitconfig`.
- After the reset, all work was committed and pushed to a **new GitHub repo**: `https://github.com/Caebyr/Capstone-New`, using a remote named **`neworigin`** (`git push neworigin main` succeeded; the last commit is "4 Oct 2026").
- VS Code's status bar shows `main` as **141 commits ahead**. `main` is probably still tracking an old remote/repo rather than `neworigin`.
- Miniconda (or Anaconda) and the NVIDIA driver should be freshly installed by the time you read this (Part 1). Confirm before relying on them.

### Project environment requirements (from the project docs)

- Use a conda environment named **`hf`**. Do **not** use any `tf` environment.
- Key packages, **pinned**:
  - `torch==2.5.1+cu118`
  - `transformers==4.49.0`. Newer versions refuse to load this project's CLIP checkpoint under torch < 2.6.
  - `open_clip_torch`
  - `scikit-learn==1.3.2`
  - `nbconvert` and `nbclient` (for running notebooks headlessly)
- GPU: **NVIDIA RTX 3050 Laptop, ~4 GB VRAM**. Keep batch sizes small (1–4) and use `torch.no_grad()` for inference.
- The project docs say not to recreate environments without a demonstrated compatibility issue. Here the old environment no longer exists, so recreating `hf` is required.

### Tasks for you

Do these in order and report after each section.

**A. Git: make `main` track Capstone-New as `origin`**
1. Run `git remote -v`, `git branch -vv` and `git status`, and explain what the 141-ahead count is relative to.
2. Propose a plan to:
   - rename `neworigin` to `origin` (if an existing `origin` points to an old repo, propose removing or renaming it rather than deleting history),
   - set `main` to track `origin/main` (`git push -u origin main` or `git branch --set-upstream-to=origin/main main`).
3. **Show me the plan and wait for my OK** before changing remotes or pushing. Never force-push.
4. Afterwards, plain `git push` and `git pull` should work against `Caebyr/Capstone-New`, and the ahead/behind count should be accurate.

**B. Recreate the `hf` conda environment**
1. Confirm `conda --version` and `nvidia-smi` work. If either fails, stop and tell me what to install.
2. Check the project's own docs (e.g. `CLAUDE.md`, `README.md`, `TECHNICAL_REPORT.md`) for a stated Python version. If none is stated, use **Python 3.11**, which is compatible with torch 2.5.1.
3. Create the environment: `conda create -n hf python=<version> -y`.
4. Install torch **first** from the CUDA 11.8 index:
   `pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
   (pick torchvision/torchaudio versions that match torch 2.5.1).
5. Install the rest **without letting pip upgrade torch or transformers** (use a constraints file or check versions afterwards):
   `transformers==4.49.0`, `open_clip_torch`, `scikit-learn==1.3.2`, `nbconvert`, `nbclient`, `ipykernel`.
6. Scan the project's `.py` and `.ipynb` files for other imports (e.g. numpy, pandas, matplotlib, Pillow) and install whatever is missing.
7. Register the kernel for notebooks: `python -m ipykernel install --user --name hf --display-name "Python (hf)"`.

**C. Verify**
1. `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` should print `2.5.1+cu118 True ...RTX 3050...`.
2. Confirm `transformers.__version__ == "4.49.0"` and `sklearn.__version__ == "1.3.2"`.
3. Load the project's CLIP checkpoint the way the project code does, to confirm it loads.
4. Run one small script or notebook end to end (e.g. `final_premst_results/make_final_figures.py`, or a short notebook via nbclient) with small batch sizes.

**D. Fix paths and make this reproducible**
1. Search the code, notebooks and configs for hardcoded `C:\Users\Kabir Oberoi\` paths. List them and propose fixes (prefer paths relative to the project root). Ask before editing.
2. Export the working environment to `environment.yml` (and/or a `requirements.txt` with the pinned versions) so next time it's one command.
3. Make sure `.gitignore` covers environment folders, caches, large checkpoints and data that shouldn't be pushed. Check sizes before adding anything big to git.

**E. VS Code**
1. Tell me how to select the `hf` interpreter (`Ctrl+Shift+P` → **Python: Select Interpreter**) and the `Python (hf)` kernel in notebooks.

### Rules
- Ask before: deleting files, changing git remotes, pushing, editing project code, or anything that can't be undone.
- Never force-push or rewrite git history.
- Don't change the pinned versions above unless something fails, and explain why if you need to.
- Keep GPU memory use small (batch sizes 1–4, `torch.no_grad()` for inference).
- When finished, give me a short summary: what was done, what still needs me, and the exact commands for my daily routine (activate env, commit, push).