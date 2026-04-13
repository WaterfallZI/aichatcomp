"""
AI Chat Pro — Telegram Bot
Uses OpenRouter API (same as web version)
Deploy: python tgbot.py
Env vars: TG_TOKEN, OPENROUTER_KEY
"""
import os, requests, json, logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

TG_TOKEN       = '8326227335:AAGKxguIxbuyBcXyXtsfDVtpbL0l4QMWOv8'
OPENROUTER_KEY = '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'
OR_URL         = 'https://openrouter.ai/api/v1/chat/completions'
DEFAULT_MODEL  = 'meta-llama/llama-3.3-70b-instruct'
MAX_HISTORY    = 20

# In-memory chat history per user
history: dict[int, list] = {}

SYSTEM_PROMPT = """You are AI Chat Pro — a helpful, accurate AI assistant.
Be concise, use markdown when helpful. Answer in the user's language."""

MODELS = {
    '1': ('meta-llama/llama-3.3-70b-instruct', 'Llama 3.3 70B'),
    '2': ('meta-llama/llama-3.1-8b-instruct',  'Llama 3.1 8B (fast)'),
    '3': ('mistralai/mixtral-8x7b-instruct',    'Mixtral 8x7B'),
    '4': ('deepseek/deepseek-chat',             'DeepSeek Chat'),
    '5': ('anthropic/claude-3-haiku',           'Claude 3 Haiku'),
    '6': ('openai/gpt-4o-mini',                 'GPT-4o Mini'),
}

user_models: dict[int, str] = {}


def tg(method: str, **kwargs) -> dict:
    r = requests.post(
        f'https://api.telegram.org/bot{TG_TOKEN}/{method}',
        json=kwargs, timeout=30
    )
    return r.json()


def send(chat_id: int, text: str, **kwargs):
    return tg('sendMessage', chat_id=chat_id, text=text,
               parse_mode='Markdown', **kwargs)


def send_typing(chat_id: int):
    tg('sendChatAction', chat_id=chat_id, action='typing')


def ask_ai(user_id: int, user_text: str) -> str:
    model = user_models.get(user_id, DEFAULT_MODEL)
    msgs  = history.get(user_id, [])

    msgs.append({'role': 'user', 'content': user_text})
    if len(msgs) > MAX_HISTORY:
        msgs = msgs[-MAX_HISTORY:]

    payload = {
        'model': model,
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}] + msgs,
        'max_tokens': 2048,
        'temperature': 0.7,
    }
    headers = {
        'Authorization': f'Bearer {OPENROUTER_KEY}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://aichatcompany.up.railway.app',
        'X-Title': 'AI Chat Pro TG Bot',
    }

    try:
        r = requests.post(OR_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        reply = r.json()['choices'][0]['message']['content']
        msgs.append({'role': 'assistant', 'content': reply})
        history[user_id] = msgs
        return reply
    except Exception as e:
        log.error(f'OpenRouter error: {e}')
        return f'❌ AI error: {e}'


def handle_update(update: dict):
    msg = update.get('message') or update.get('edited_message')
    if not msg:
        return

    chat_id = msg['chat']['id']
    user_id = msg['from']['id']
    text    = msg.get('text', '').strip()

    if not text:
        return

    # Commands
    if text == '/start':
        model_name = dict(MODELS.values()).get(user_models.get(user_id, DEFAULT_MODEL), 'Llama 3.3 70B')
        send(chat_id, (
            "👋 *Welcome to AI Chat Pro!*\n\n"
            "I'm powered by multiple AI models via OpenRouter.\n\n"
            f"Current model: `{user_models.get(user_id, DEFAULT_MODEL)}`\n\n"
            "Commands:\n"
            "/model — change AI model\n"
            "/clear — clear chat history\n"
            "/help — show help\n\n"
            "Just send me any message to start chatting!"
        ))
        return

    if text == '/help':
        send(chat_id, (
            "🤖 *AI Chat Pro Help*\n\n"
            "Just type any message and I'll respond.\n\n"
            "*Commands:*\n"
            "/start — welcome message\n"
            "/model — choose AI model\n"
            "/clear — clear conversation history\n"
            "/help — this message\n\n"
            "*Tips:*\n"
            "• I remember your conversation history\n"
            "• Use /clear to start fresh\n"
            "• Switch models with /model"
        ))
        return

    if text == '/clear':
        history.pop(user_id, None)
        send(chat_id, "🗑️ Conversation cleared. Starting fresh!")
        return

    if text == '/model':
        keyboard = {
            'inline_keyboard': [
                [{'text': f"{'✅ ' if user_models.get(user_id, DEFAULT_MODEL) == m else ''}{name}", 
                  'callback_data': f'model:{k}'}]
                for k, (m, name) in MODELS.items()
            ]
        }
        send(chat_id, "🤖 *Choose AI model:*", reply_markup=keyboard)
        return

    # Regular message — ask AI
    send_typing(chat_id)
    reply = ask_ai(user_id, text)

    # Telegram markdown can fail on complex text — fallback to plain
    try:
        send(chat_id, reply)
    except Exception:
        tg('sendMessage', chat_id=chat_id, text=reply)


def handle_callback(callback: dict):
    query_id = callback['id']
    user_id  = callback['from']['id']
    chat_id  = callback['message']['chat']['id']
    data     = callback.get('data', '')

    if data.startswith('model:'):
        key = data.split(':')[1]
        if key in MODELS:
            model, name = MODELS[key]
            user_models[user_id] = model
            tg('answerCallbackQuery', callback_query_id=query_id,
               text=f'✅ Switched to {name}')
            send(chat_id, f"✅ Model changed to *{name}*\n`{model}`")
    else:
        tg('answerCallbackQuery', callback_query_id=query_id)


def run_polling():
    log.info('Starting AI Chat Pro Telegram Bot...')
    if not TG_TOKEN:
        log.error('TG_TOKEN not set!')
        return
    if not OPENROUTER_KEY:
        log.error('OPENROUTER_KEY not set!')
        return

    offset = 0
    log.info('Bot is running. Press Ctrl+C to stop.')

    while True:
        try:
            resp = tg('getUpdates', offset=offset, timeout=30, allowed_updates=['message', 'callback_query'])
            updates = resp.get('result', [])

            for update in updates:
                offset = update['update_id'] + 1
                try:
                    if 'callback_query' in update:
                        handle_callback(update['callback_query'])
                    else:
                        handle_update(update)
                except Exception as e:
                    log.error(f'Update error: {e}')

        except KeyboardInterrupt:
            log.info('Bot stopped.')
            break
        except Exception as e:
            log.error(f'Polling error: {e}')
            import time; time.sleep(5)


if __name__ == '__main__':
    run_polling()
