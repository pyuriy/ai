# How to Create or Open a New Google Colab Notebook

Google Colab (Colaboratory) is a free, cloud-based Jupyter notebook environment that requires no local setup—just a web browser and a Google account. Notebooks are automatically saved to your Google Drive in a "Colab Notebooks" folder. Below are step-by-step instructions based on the official documentation. These steps are current as of October 2025 and haven't changed significantly from prior versions.

## Prerequisites
- A Google account (sign in at [colab.research.google.com](https://colab.research.google.com/)).
- A modern web browser (Chrome recommended for best performance).

## Creating a New Colab Notebook
There are two main ways: directly from the Colab site or via Google Drive.

### Method 1: From the Colab Website (Quickest)
1. Open your browser and navigate to [colab.research.google.com](https://colab.research.google.com/).
2. If prompted, sign in with your Google account.
3. On the welcome screen, click the **New notebook** button (usually in the bottom-right corner or under the "Get started" section).
4. A new untitled notebook (e.g., "Untitled0.ipynb") will open in a new tab, ready for code or text cells. It's automatically saved to your Google Drive.

### Method 2: From Google Drive
1. Go to [drive.google.com](https://drive.google.com/) and sign in.
2. In the left sidebar, click **+ New** > **More** > **Google Colaboratory** (or search for "Colab" in the New menu).
3. A new untitled notebook opens directly in Colab.

**Tips**:
- Rename the notebook by clicking the title at the top (e.g., "My First Notebook").
- Add cells with **+ Code** or **+ Text** buttons in the toolbar.
- To use GPU/TPU: Go to **Runtime** > **Change runtime type** > Select hardware accelerator > Save.

## Opening an Existing Colab Notebook
Existing notebooks (.ipynb files) are stored in Google Drive. You can open them from Colab, Drive, or by URL.

### Method 1: From the Colab Interface
1. Go to [colab.research.google.com](https://colab.research.google.com/) and sign in.
2. Click the **Colab logo** (top-left) or go to **File** > **Open notebook**.
3. In the file browser:
   - Select **Google Drive** tab to browse your notebooks.
   - Or use **Upload** to select a local .ipynb file.
   - Search for recent notebooks if needed.
4. Double-click the notebook to open it in a new tab.

### Method 2: From Google Drive
1. Go to [drive.google.com](https://drive.google.com/).
2. Locate your .ipynb file (look for the yellow "CO" icon in the "Colab Notebooks" folder).
3. Right-click the file > **Open with** > **Google Colaboratory** (or double-click and select it from the top bar).
4. The notebook opens in Colab.

### Method 3: By URL or Shared Link
- If you have a shared link (e.g., from a collaborator), click it—it opens a copy in your Drive (use **File** > **Save a copy in Drive** to edit without affecting the original).
- For GitHub repos: In Colab, go to **File** > **Open notebook** > **GitHub** tab > Paste the repo URL and select the .ipynb file.

**Tips**:
- Colab autosaves changes, but download via **File** > **Download .ipynb** for backups.
- If the notebook doesn't load, ensure the file is accessible and try refreshing.
- For enterprise/Vertex AI versions, use the Google Cloud console under "Colab Enterprise" > **New notebook**.

For more details, check the [official Colab FAQ](https://research.google.com/colaboratory/faq.html). If you encounter issues, ensure your browser allows pop-ups for Drive/Colab.
