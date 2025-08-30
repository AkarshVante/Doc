import streamlit as st
import shutil
from config import FAISS_DIR, init_config
from parsers import get_pdf_text, get_text_chunks
from embeddings import build_vector_store, load_vector_store
from qa_chain import (build_plain_prompt, build_bullets_prompt,
                      generate_answer_with_fallback_using_prompt)
from utils import (
    init_session_state, add_message, find_preceding_user_message_text,
    render_markdown_like_to_html, format_time
)
import datetime
import html as html_module
import streamlit.components.v1 as components

# initialize configuration (loads env / configures genai)
init_config()

# Streamlit app
def main():
    st.set_page_config(page_title="ChatPDF — Gemini (modular)", layout="wide")
    init_session_state()

    # Header
    cols = st.columns([0.75, 0.25])
    with cols[0]:
        st.title("💻 ChatPDF — Plain & Bullets")
        st.caption("Upload PDFs, process them and chat — choose Plain or Bullets formatting.")
    with cols[1]:
        st.metric(label="", value="Yes" if st.session_state.faiss_ready else "No")

    left_col, center_col, right_col = st.columns([1.5, 3, 1])

    # Left: conversations + export
    with left_col:
        st.header("💬 Conversations")
        if not st.session_state.history:
            st.info("No messages yet — your conversation history will appear here.")
        else:
            preview_items = []
            for i, m in enumerate(st.session_state.history):
                role = "You" if m['role'] == 'user' else "Assistant"
                preview = m['text'][:80].replace('\n', ' ')
                ts = format_time(m.get('time', ''))
                preview_items.append(f"{i}: [{ts}] {role}: {preview}")

            default_index = 0
            if st.session_state.get("focus_index") is not None:
                fi = st.session_state.focus_index
                if 0 <= fi < len(preview_items):
                    default_index = fi

            selected = st.radio("Select message to focus", options=list(range(len(preview_items))), index=default_index, format_func=lambda x: preview_items[x])
            if st.session_state.get("focus_index") != selected:
                st.session_state.focus_index = selected

            cols_left_actions = st.columns([0.5, 0.5])
            with cols_left_actions[0]:
                if st.button("Clear History"):
                    st.session_state.history = []
                    st.session_state.focus_index = None
                    st.success("Chat history cleared.")
            with cols_left_actions[1]:
                st.write(" ")

            st.markdown("---")
            st.subheader("Export")
            conv_text = []
            for m in st.session_state.history:
                role = "You" if m['role'] == 'user' else 'Assistant'
                conv_text.append(f"[{format_time(m.get('time',''))}] {role}: {m['text']}")
            if conv_text:
                st.download_button("Download conversation", data="\n".join(conv_text), file_name="chatpdf_conversation.txt")
            else:
                st.info("Nothing to download yet.")

    # Center: Input form and chat UI
    with center_col:
        with st.form(key="question_form", clear_on_submit=True):
            user_question = st.text_area("", placeholder="Type your question and press Send...", height=120, key='chat_input')
            format_choice = st.selectbox("Answer style", options=["Plain text", "Bullets"], index=0)
            cols_btn = st.columns([0.15, 0.85])
            with cols_btn[0]:
                send_btn = st.form_submit_button("Send")
            with cols_btn[1]:
                st.write(" ")

            if send_btn and user_question and user_question.strip():
                if not st.session_state.faiss_ready:
                    st.error("FAISS index not found. Please upload and process PDFs first.")
                else:
                    add_message('user', user_question)
                    try:
                        db = load_vector_store()
                        docs = db.similarity_search(user_question, k=4)
                    except Exception as e:
                        st.error(f"Failed to load FAISS index: {e}")
                        docs = []

                    if docs:
                        with st.spinner("Generating answer..."):
                            prompt_template = build_plain_prompt() if format_choice == "Plain text" else build_bullets_prompt()
                            answer_text, model_used, error = generate_answer_with_fallback_using_prompt(prompt_template, docs, user_question)

                        if answer_text:
                            add_message('assistant', answer_text)
                            st.session_state.last_model_used = model_used
                            st.session_state.focus_index = len(st.session_state.history) - 1
                            st.success("Answer generated and appended to conversation.")
                        else:
                            st.error(f"Failed to generate answer: {error}")

        # Upload expander
        with st.expander("📎 Upload PDFs (attach & process here)"):
            uploaded = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])
            if st.button("Submit & Process Files"):
                if not uploaded:
                    st.warning("Please upload one or more PDF files first.")
                else:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)

                    progress_text.info("Stage 1/3 — Extracting text from PDFs...")
                    raw_text = get_pdf_text(uploaded)
                    progress_bar.progress(10)

                    if not raw_text.strip():
                        st.error("No readable text found in the uploaded PDFs.")
                    else:
                        progress_text.info("Stage 2/3 — Chunking text for embeddings...")
                        text_chunks = get_text_chunks(raw_text)
                        progress_bar.progress(40)

                        progress_text.info("Stage 3/3 — Creating embeddings and saving FAISS index...")

                        def cb(msg):
                            progress_text.info(msg)

                        build_vector_store(text_chunks, progress_callback=cb)
                        progress_bar.progress(100)

                        st.session_state.faiss_ready = True
                        st.success("✅ Processing complete — FAISS index saved.")

        # Render chat window
        chat_box = st.container()
        st.markdown("""                    <style>
            .chat-window { max-height: 65vh; overflow: auto; padding: 12px; background: #ffffff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
            .message { margin: 8px 0; display: flex; align-items: flex-end; }
            .bubble { max-width: 78%; padding: 12px 14px; border-radius: 12px; line-height: 1.4; }
            .bubble.user { background: linear-gradient(90deg,#2b90ff,#1e6fe8); color: white; border-bottom-right-radius: 6px; margin-left: auto; }
            .bubble.assistant { background: #f1f3f5; color: #111827; border-bottom-left-radius: 6px; margin-right: auto; }
            .meta { font-size: 11px; color: #6b7280; margin-top: 4px; }
            .focused { box-shadow: 0 0 0 3px rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.4); border-radius: 12px; padding: 10px; background: linear-gradient(90deg, #fffefc, #f7fbff); }
            .bubble ul { margin: 8px 0 8px 18px; }
            .bubble ol { margin: 8px 0 8px 18px; }
            pre { background: #0b1220; color: #e6edf3; padding: 10px; border-radius: 8px; overflow: auto; }
            </style>
            """, unsafe_allow_html=True)

        chat_html = "<div class='chat-window' id='chat-window'>"
        if not st.session_state.history:
            chat_html += "<div style='padding:20px;color:#6b7280'>No messages yet — upload PDFs and ask a question!</div>"
        else:
            for idx, msg in enumerate(st.session_state.history):
                ts = format_time(msg.get('time',''))
                msg_id = f"msg-{idx}"
                focused_class = "focused" if st.session_state.focus_index is not None and st.session_state.focus_index == idx else ""

                if msg['role'] == 'assistant':
                    content_html = render_markdown_like_to_html(msg['text'])
                    chat_html += (
                        f"<div class='message' id='{msg_id}'>"
                        f"<div class='bubble assistant {focused_class}'>{content_html}<div class='meta'>{ts}</div></div>"
                        f"</div>"
                    )
                else:
                    content_html = "<p>" + html_module.escape(msg['text']).replace("\n","<br/>") + "</p>"
                    chat_html += (
                        f"<div class='message' id='{msg_id}'>"
                        f"<div class='bubble user {focused_class}'>{content_html}<div class='meta'>{ts}</div></div>"
                        f"</div>"
                    )
        chat_html += "</div>"
        chat_box.markdown(chat_html, unsafe_allow_html=True)

        if st.session_state.focus_index is not None:
            focus_idx = st.session_state.focus_index
            if 0 <= focus_idx < len(st.session_state.history):
                scroll_script = f"""
                <script>
                const el = document.getElementById("msg-{focus_idx}");
                if (el) {{ 
                    el.scrollIntoView({{behavior: "smooth", block: "center"}});
                }}
                </script>
                """
                components.html(scroll_script, height=0, width=0)

    # Right: controls + regenerate
    with right_col:
        st.header("Controls")
        st.markdown("**FAISS status:**")
        if st.session_state.faiss_ready:
            st.success("FAISS index available.")
        else:
            st.warning("No FAISS index found.")

        st.markdown("---")
        st.subheader("Reformat Answer")

        focus_idx = st.session_state.focus_index
        target_idx = None
        if focus_idx is not None and 0 <= focus_idx < len(st.session_state.history):
            if st.session_state.history[focus_idx]["role"] == "assistant":
                target_idx = focus_idx
            else:
                for j in range(focus_idx+1, len(st.session_state.history)):
                    if st.session_state.history[j]["role"] == "assistant":
                        target_idx = j
                        break
        else:
            for j in range(len(st.session_state.history)-1, -1, -1):
                if st.session_state.history[j]["role"] == "assistant":
                    target_idx = j
                    break

        if target_idx is not None:
            st.write(f"Selected assistant message index: {target_idx}")
            st.write("Regenerate this answer in a new format (adds a new assistant message).")
            cols_regen = st.columns([1,1])
            with cols_regen[0]:
                if st.button("Regenerate — Plain Text", key=f"regen_plain_{target_idx}"):
                    user_q = find_preceding_user_message_text(target_idx)
                    if not user_q:
                        st.error("Could not locate the original user question to regenerate.")
                    else:
                        try:
                            db = load_vector_store()
                            docs = db.similarity_search(user_q, k=4)
                        except Exception as e:
                            st.error(f"Failed to load FAISS index: {e}")
                            docs = []

                        if docs:
                            with st.spinner("Regenerating (plain text)..."):
                                prompt_template = build_plain_prompt()
                                answer_text, model_used, error = generate_answer_with_fallback_using_prompt(prompt_template, docs, user_q)
                            if answer_text:
                                add_message('assistant', answer_text)
                                st.session_state.last_model_used = model_used
                                st.session_state.focus_index = len(st.session_state.history) - 1
                                st.success("Regenerated (plain text)")
                            else:
                                st.error(f"Regeneration failed: {error}")
            with cols_regen[1]:
                if st.button("Regenerate — Bullets", key=f"regen_bullets_{target_idx}"):
                    user_q = find_preceding_user_message_text(target_idx)
                    if not user_q:
                        st.error("Could not locate the original user question to regenerate.")
                    else:
                        try:
                            db = load_vector_store()
                            docs = db.similarity_search(user_q, k=4)
                        except Exception as e:
                            st.error(f"Failed to load FAISS index: {e}")
                            docs = []

                        if docs:
                            with st.spinner("Regenerating (bullets)..."):
                                prompt_template = build_bullets_prompt()
                                answer_text, model_used, error = generate_answer_with_fallback_using_prompt(prompt_template, docs, user_q)
                            if answer_text:
                                add_message('assistant', answer_text)
                                st.session_state.last_model_used = model_used
                                st.session_state.focus_index = len(st.session_state.history) - 1
                                st.success("Regenerated (bullets)." )
                            else:
                                st.error(f"Regeneration failed: {error}")
        else:
            st.info("No assistant message available to regenerate. Send a question first.")

        st.markdown("---")
        st.subheader("File Uploads")
        if st.button("Delete All"):
            try:
                if __import__('os').path.isdir(FAISS_DIR):
                    shutil.rmtree(FAISS_DIR)
                st.session_state.faiss_ready = False
                st.success("FAISS index deleted.")
            except Exception as e:
                st.error(f"Failed to delete FAISS index: {e}")

        st.markdown("---")
        if st.session_state.last_model_used:
            st.write(f"_Last model used: {st.session_state.last_model_used}_")

if __name__ == '__main__':
    main()
