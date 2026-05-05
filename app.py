"""
Gradio interface with multi-provider support.
"""
import gradio as gr

from config import (
    validate_config,
    get_available_providers,
    PROVIDERS,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
)
from src.pipeline import YouTubeBot


# Validate at startup
validate_config()
available_providers = get_available_providers()

bot = YouTubeBot()


# ---------- Handlers ----------
def handle_provider_change(provider: str):
    """When provider changes, update the model dropdown."""
    models = PROVIDERS[provider]["models"]
    return gr.update(choices=models, value=models[0])


def handle_llm_switch(provider: str, model: str):
    try:
        return bot.set_llm(provider, model)
    except Exception as e:
        return f"❌ Error: {e}"


def handle_load(url: str):
    try:
        return bot.load_video(url.strip())
    except Exception as e:
        return f"❌ Error: {e}"


def handle_summarize(url: str, provider: str, model: str):
    try:
        bot.set_llm(provider, model)
        if url.strip() != bot.current_url:
            bot.load_video(url.strip())
        return bot.summarize()
    except Exception as e:
        return f"❌ Error: {e}"


def handle_question(url: str, question: str, provider: str, model: str):
    try:
        bot.set_llm(provider, model)
        if url.strip() != bot.current_url:
            bot.load_video(url.strip())
        return bot.ask(question.strip())
    except Exception as e:
        return f"❌ Error: {e}"


# ---------- UI ----------
with gr.Blocks(title="YouTube QA Bot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 YouTube Video Summarizer & Q&A")
    gr.Markdown("Multi-provider support: AgentRouter + OpenRouter")

    # ===== Provider/Model Selection =====
    with gr.Group():
        gr.Markdown("### ⚙️ LLM Settings")
        with gr.Row():
            provider_dropdown = gr.Dropdown(
                label="Provider",
                choices=available_providers,
                value=DEFAULT_PROVIDER if DEFAULT_PROVIDER in available_providers else available_providers[0],
                interactive=True,
                scale=1,
            )
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=PROVIDERS[DEFAULT_PROVIDER]["models"]
                if DEFAULT_PROVIDER in available_providers
                else PROVIDERS[available_providers[0]]["models"],
                value=DEFAULT_MODEL,
                interactive=True,
                scale=2,
            )
            llm_status = gr.Textbox(
                label="LLM Status",
                interactive=False,
                scale=2,
            )

    # ===== Video Loading =====
    with gr.Group():
        gr.Markdown("### 📺 Video")
        with gr.Row():
            url_input = gr.Textbox(
                label="YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                scale=4,
            )
            load_btn = gr.Button("📥 Load Video", scale=1, variant="primary")

        video_status = gr.Textbox(label="Video Status", interactive=False)

    # ===== Tabs =====
    with gr.Tab("📝 Summary"):
        summarize_btn = gr.Button("Summarize", variant="primary")
        summary_output = gr.Textbox(label="Summary", lines=10)

    with gr.Tab("❓ Q&A"):
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="What is this video about?",
            lines=2,
        )
        ask_btn = gr.Button("Ask", variant="primary")
        answer_output = gr.Textbox(label="Answer", lines=10)

    # ---------- Wiring ----------
    # When provider changes, update available models
    provider_dropdown.change(
        handle_provider_change,
        inputs=provider_dropdown,
        outputs=model_dropdown,
    )

    # When model changes, switch the LLM
    model_dropdown.change(
        handle_llm_switch,
        inputs=[provider_dropdown, model_dropdown],
        outputs=llm_status,
    )

    # Video loading
    load_btn.click(handle_load, inputs=url_input, outputs=video_status)

    # Summarization (passes provider+model to ensure latest selection)
    summarize_btn.click(
        handle_summarize,
        inputs=[url_input, provider_dropdown, model_dropdown],
        outputs=summary_output,
    )

    # Q&A
    ask_btn.click(
        handle_question,
        inputs=[url_input, question_input, provider_dropdown, model_dropdown],
        outputs=answer_output,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)