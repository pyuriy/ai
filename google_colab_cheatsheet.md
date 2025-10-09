# Comprehensive Cheat Sheet for Google Colab Notebooks

Google Colab is a free, cloud-based Jupyter notebook environment hosted by Google, ideal for Python coding, data science, ML, and education. It offers free GPU/TPU access, seamless Google Drive integration, and real-time collaboration. No local setup needed—just a browser and Google account. This cheat sheet covers essentials, shortcuts, tips, and advanced usage as of October 2025. It's Jupyter-compatible but highlights Colab-specific features.

## 🚀 Getting Started
- **Access**: Go to [colab.research.google.com](https://colab.research.google.com) and sign in. Click **New notebook** for a blank one, or **File > Open notebook** to upload/browse.
- **Rename/Save**: Click the title to rename; autosaves to Google Drive in "Colab Notebooks" folder. Download via **File > Download .ipynb**.
- **Cells**: Add with **+ Code** (Python) or **+ Text** (Markdown). Run with Shift+Enter or toolbar play button.
- **Runtime Setup**: **Runtime > Change runtime type** > Select Python 3, GPU/TPU, or high-RAM. Free tier: Up to 12-hour sessions; Pro: 24 hours.
- **Install Packages**: Use `!pip install package` in a code cell (e.g., `!pip install transformers`). Persists per session.

**Pro Tip**: Mount Drive for file access: `from google.colab import drive; drive.mount('/content/drive')`.

## ⌨️ Keyboard Shortcuts
Colab supports standard Jupyter shortcuts. View full list: **Help > Keyboard Shortcuts**. Here's a curated table of essentials (Command on Mac, Ctrl on Windows/Linux).

| Category       | Shortcut                  | Action |
|----------------|---------------------------|--------|
| **Editing**   | A (above), B (below)     | Insert cell above/below |
|                | M                        | Change to Markdown (text) |
|                | Y                        | Change to Code |
|                | D, D (twice)             | Delete selected cell |
|                | Ctrl+M H                 | Toggle line numbers |
| **Running**   | Shift+Enter              | Run cell & select next |
|                | Ctrl+Enter               | Run cell & stay |
|                | Ctrl+F9                  | Run all cells |
|                | Ctrl+M .                 | Run cell & advance |
| **Selection** | Shift+↑/↓                | Select multiple cells |
|                | Ctrl+M A                 | Merge above |
|                | Ctrl+M J                 | Merge below |
| **Navigation**| Ctrl+↑/↓                 | Jump to cell boundary |
|                | Ctrl+Home/End            | Go to notebook start/end |
| **Undo/Redo** | Ctrl+Z, Ctrl+Y           | Undo/Redo |
| **Other**     | Ctrl+/                   | Comment/uncomment |
|                | Ctrl+Shift+P             | Command palette (search commands) |

**Practice Tip**: Use in command mode (click cell margin or Esc). Edit mode: Enter.

## 📝 Cell Management & Markdown
- **Cell Types**: Code for Python; Markdown for formatted text (e.g., # Heading, *italic*, **bold**, [link](url), `inline code`).
- **Markdown Tips**:
  - Lists: - Bullet or 1. Numbered.
  - Tables: \| Header1 \| Header2 \| \n \|---\|---\| \n \| Data1 \| Data2 \|
  - Images: ![alt](url) or from Drive: `/content/drive/MyDrive/image.png`.
  - LaTeX: $E=mc^2$ inline, $$ equation $$ block.
- **Forms**: Add interactive inputs: **Insert > Add form field** for variables.
- **Output Control**: Clear: **Edit > Clear cell output**. Hide: **View > Show/hide code**.

**Example Markdown Cell**:
```
# Title
This is **bold** and a [link](https://colab.research.google.com).

- Item 1
- Item 2
```

## ⚙️ Runtime & Resources
- **Hardware**: Free: T4 GPU (limited); Pro: A100/V100. Check usage: **Runtime > View resources**.
- **Sessions**: Idle timeout ~90 min; max 12h free, 24h Pro. Reset: **Runtime > Disconnect and delete runtime** (limited frequency).
- **Magic Commands** (% for line, %% for cell):
  | Command | Description |
  |---------|-------------|
  | %timeit | Benchmark code |
  | %matplotlib inline | Inline plots |
  | %%shell | Run shell commands |
  | %load_ext tensorboard | Load TensorBoard |
  | !ls /content | List files |
  | %debug | Enter debugger |
- **Environment**: Python 3.10+; pre-installed: NumPy, Pandas, Matplotlib, TensorFlow, PyTorch.
- **Limits**: No Python 2; dynamic quotas (e.g., ~12GB RAM free). Avoid disallowed activities (e.g., mining, SSH) to prevent bans.

**Tip**: For long runs, use checkpoints: Save models to Drive.

## 👥 Collaboration & Sharing
- **Share**: Click **Share** button > Add emails/roles (Viewer/Editor/Commenter). Like Google Docs—real-time edits.
- **Comments**: Highlight text > Add comment. Resolve via sidebar.
- **Version History**: **File > Revision history** (like Drive).
- **Embed/Public**: **Share > Get link** > Change to "Anyone with link".
- **Note**: Outputs/installs not shared—recipients rerun cells.

**Pro Tip**: Omit outputs before sharing: **Edit > Notebook settings > Omit code cell output**.

## 🔗 Integrations
- **Google Drive**: Mount as above; access via `/content/drive/MyDrive/`. Quota: 15GB free; avoid root folder overload (use subfolders).
- **GitHub**: **File > Open notebook > GitHub** > Paste repo URL. Save back: **File > Save a copy in GitHub**.
- **Upload Files**: **Files** sidebar > Upload icon (drag-drop). Or `from google.colab import files; files.upload()`.
- **Local Runtime**: **Connect > Connect to local runtime** for custom setups (requires Jupyter server).

## 🛠️ Best Practices & Tips
- **Efficiency**: Close unused tabs; match runtime to needs (CPU for light tasks). Use `%%capture` to suppress output.
- **Debugging**: `%pdb on` for post-mortem; `!nvidia-smi` for GPU stats.
- **Data Handling**: For large files, use Drive or `wget` (e.g., `!wget url`). Compress: `!zip -r archive.zip folder`.
- **Security**: Review AI-generated code; avoid sensitive data in shared notebooks.
- **Customization**: Browser fonts via settings; themes via extensions (e.g., Colab Dark Theme).
- **AI Features** (Gemini, for 18+ in supported regions): Spark icon for code gen/fixing; Data Science Agent for autonomous analysis. Review outputs.

| Tip Category | Examples |
|--------------|----------|
| **Performance** | Preload data in setup cell; use `torch.cuda.empty_cache()` for GPU. |
| **Documentation** | Use Markdown headers; add !git for version control. |
| **Collaboration** | Tag @user in comments; use sections for organization. |

## ⚠️ Common Issues & Solutions
| Issue | Solution |
|-------|----------|
| **Runtime Disconnects** | Stay active; upgrade to Pro for longer sessions. |
| **Drive Mount Fails** | Check quota; use subfolders; retry after 24h for errors. |
| **GPU Unavailable** | Switch to None; try later or Pro. |
| **I/O Errors** | Empty Trash; avoid >10K items in folder. |
| **AI Not Working** | Check age/locale; send feedback via Help. |
| **Slow Upload** | Use Drive links or gsutil for GCS. |

**Disallowed**: Crypto mining, deepfakes, multi-account abuse—leads to termination.

## 💡 Advanced Features
- **Colab Pro/Pro+**: $9.99–$49.99/mo; higher RAM/GPU priority, background execution. Pro for Education: Free for verified students.
- **Enterprise**: GCP integration for teams; no limits.
- **Extensions**: Install via `!pip` (e.g., nbconvert for PDF export).
- **Export**: **File > Download > .ipynb, .py, PDF**.

For more, explore official intro notebook or community cheat sheets. Experiment in a new notebook!
