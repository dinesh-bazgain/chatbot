css = """
<style>
    body {
        background: #0e0e11;
        font-family: 'Fira Code', monospace;
        margin: 0;
        padding: 2rem;
        color: #f5f5f5;
        display: flex;
        justify-content: center;
    }

    .chat-box {
        background: #1a1a1f;
        padding: 1.5rem;
        border-radius: 12px;
        width: 100%;
        max-width: 640px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .human-template,
    .bot-template {
        padding: 1rem;
        border-radius: 10px;
        font-size: 15px;
        line-height: 1.5;
        position: relative;
    }

    .human-template {
        background-color: #2f3e75;
        align-self: flex-start;
    }

    .bot-template {
        background-color: #1d5c3b;
        align-self: flex-end;
    }

    .label {
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 0.3rem;
        opacity: 0.7;
    }

    .human-template .label {
        color: #9ec4ff;
    }

    .bot-template .label {
        color: #a2ffd5;
    }
</style>
"""


user_template = """
<div class="chat-box">
    <div class="human-template">
        <span class="label">Human: </span>
        {{ question }}
    </div>
</div>
"""

bot_template = """
<div class="chat-box">
    <div class="bot-template">
        <span class="label">Bot: </span>
        {{ answer }}
    </div>
</div>
"""