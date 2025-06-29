# Chatwoot ↔️ Custom ChatGPT Q&A Connector

A connector between Chatwoot and a custom-prompted ChatGPT Q&A system.

---

## 🚧 Current Limitations

- **"Touching" people functionality** is commented out, as automatic pop-up is not possible without access to website code. Users must open the chat manually.
- **Server hosting** (Render, free subscription) may go to sleep after inactivity, causing the first request to be slow.

---

## 🐞 Reporting Issues & Requesting Changes

1. Go to the main page of the repo (where this README is located) using the link provided.
2. Click **Issues** (top left corner).
3. Click the green **New issue** button (top right).
4. Enter an intuitive title.
5. Write a clear description of the problem or change you want.
6. (Optional) Attach screenshots (drag and drop).

---

## 📝 Prompt Engineering: How to Edit the System Prompt

1. Open the `main.py` file.
2. Click the pencil icon (**Edit this file**).
3. Search for `system_prompt` (use Ctrl+F).
4. Adjust the prompt as you wish.  
   *Tip: You can ask ChatGPT to help rewrite your prompt, e.g., "Here is my prompt for my AI assistant, adjust the prompt so that replies are more friendly and elaborate."*
5. Click the green **Commit changes** button (top right).
6. Enter a short, intuitive commit message (e.g., "adjusted prompt").
7. Click **Commit changes**.

---

## 📄 Uploading New PDF Files for RAG

1. Go to the main page of the repo (where this README is located).
2. Click the **docs** folder.
3. Click **Add file** (top right).
4. Select **Upload files** in the popup.
5. In the "Add files via upload" text field, write a short, intuitive title explaining what file you added.
6. Click the green **Commit changes** button.

---

## 🤖 Changing the GPT Model

1. Open the `main.py` file.
2. Click the pencil icon (**Edit this file**).
3. Search for `MODEL_NAME` (use Ctrl+F; it's near the top of the file).
4. Enter the desired model name.  
   *Tip: You can ask ChatGPT for model recommendations, names, and pricing details.*
5. Click the green **Commit changes** button (top right).
6. Enter a short, intuitive commit message (e.g., "changed model").
7. Click **Commit changes**.

---