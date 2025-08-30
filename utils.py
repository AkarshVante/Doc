import streamlit as st
import os
import datetime
import html as html_module
import re

from config import FAISS_DIR

def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "faiss_ready" not in st.session_state:
        st.session_state.faiss_ready = os.path.isdir(FAISS_DIR)
    if "last_model_used" not in st.session_state:
        st.session_state.last_model_used = None
    if "focus_index" not in st.session_state:
        st.session_state.focus_index = None

def add_message(role, text):
    st.session_state.history.append({
        "role": role,
        "text": text,
        "time": datetime.datetime.now().isoformat()
    })

def find_preceding_user_message_text(idx):
    for i in range(idx - 1, -1, -1):
        if st.session_state.history[i]["role"] == "user":
            return st.session_state.history[i]["text"]
    return None

def format_time(iso_ts: str) -> str:
    try:
        dt = datetime.datetime.fromisoformat(iso_ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            cleaned = iso_ts.rstrip("Z")
            dt = datetime.datetime.fromisoformat(cleaned)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return iso_ts

def render_markdown_like_to_html(text: str) -> str:
    if not text:
        return ""
    escaped = html_module.escape(text)
    lines = escaped.splitlines()
    html_parts = []
    i = 0
    in_ul = False
    in_ol = False
    in_code = False
    code_lang = ""
    code_lines = []

    while i < len(lines):
        line = lines[i]
        if not in_code and line.strip().startswith("```"):
            in_code = True
            code_lang = line.strip()[3:].strip()
            code_lines = []
            i += 1
            continue
        if in_code:
            if line.strip().startswith("```"):
                code_html = "<pre><code"
                if code_lang:
                    code_html += f" class='lang-{html_module.escape(code_lang)}'"
                code_html += ">" + "\n".join(code_lines) + "</code></pre>"
                html_parts.append(code_html)
                in_code = False
                code_lang = ""
                code_lines = []
                i += 1
                continue
            else:
                code_lines.append(line)
                i += 1
                continue

        if re.match(r"^#{1,6}\s+", line):
            level = len(re.match(r"^(#+)", line).group(1))
            content = line[level+1:].strip()
            html_parts.append(f"<h{level}>{content}</h{level}>")
            i += 1
            continue

        if re.match(r"^\s*([-*])\s+", line):
            if not in_ul:
                in_ul = True
                html_parts.append("<ul>")
            content = re.sub(r"^\s*([-*])\s+", "", line)
            html_parts.append(f"<li>{content}</li>")
            i += 1
            if i < len(lines):
                if not re.match(r"^\s*([-*])\s+", lines[i]):
                    html_parts.append("</ul>")
                    in_ul = False
            else:
                html_parts.append("</ul>")
                in_ul = False
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            if not in_ol:
                in_ol = True
                html_parts.append("<ol>")
            content = re.sub(r"^\s*\d+\.\s+", "", line)
            html_parts.append(f"<li>{content}</li>")
            i += 1
            if i < len(lines):
                if not re.match(r"^\s*\d+\.\s+", lines[i]):
                    html_parts.append("</ol>")
                    in_ol = False
            else:
                html_parts.append("</ol>")
                in_ol = False
            continue

        if line.strip() == "":
            html_parts.append("<br/>")
            i += 1
            continue

        para_lines = [line]
        j = i + 1
        while j < len(lines) and lines[j].strip() != "" and not re.match(r"^\s*([-*])\s+", lines[j]) and not re.match(r"^\s*\d+\.\s+", lines[j]) and not re.match(r"^#{1,6}\s+", lines[j]) and not lines[j].strip().startswith("```"):
            para_lines.append(lines[j])
            j += 1
        paragraph = " ".join([l.strip() for l in para_lines])
        html_parts.append(f"<p>{paragraph}</p>")
        i = j

    if in_ul:
        html_parts.append("</ul>")
    if in_ol:
        html_parts.append("</ol>")
    return "\n".join(html_parts)
