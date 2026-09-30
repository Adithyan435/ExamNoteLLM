import os
from threading import Thread
import telebot
from transformers import TextIteratorStreamer

def launch_telegram_bot(assistant, bot_token: str = None):
    token = bot_token or os.getenv("TELEGRAM_API_KEY", "8357868813:AAFeDCX3E1sQqFAbOoAd-LPBVVCe8YZGAHI")
    bot = telebot.TeleBot(token)
    user_profiles = {}
    user_flows = {}

    def get_profile(chat_id):
        if chat_id not in user_profiles:
            user_profiles[chat_id] = {'subject': 'General', 'institution': 'Universal', 'level': 'Undergraduate'}
        return user_profiles[chat_id]

    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(msg):
        prof = get_profile(msg.chat.id)
        bot.reply_to(msg, f"Academic Assistant Online\nProfile: {prof['subject']} | {prof['institution']} | {prof['level']}\n\nCommands:\n/ask - Syllabus notes\n/eval - Answer grading\n/quiz - Question generation\n/notes - Revision sheet\n/profile - Set profile\n/cancel - Abort")

    @bot.message_handler(commands=['cancel'])
    def handle_cancel(msg):
        if msg.chat.id in user_flows:
            del user_flows[msg.chat.id]
            bot.reply_to(msg, "Session cancelled.")
        else:
            bot.reply_to(msg, "No active session.")

    @bot.message_handler(commands=['ask'])
    def handle_ask(msg):
        user_flows[msg.chat.id] = {'action': 'ask'}
        bot.reply_to(msg, "Enter exam question or topic:")
        bot.register_next_step_handler(msg, step_ask_q)

    def step_ask_q(msg):
        if msg.text.startswith('/'): return
        user_flows[msg.chat.id]['q'] = msg.text
        bot.send_message(msg.chat.id, "Specify marks (2, 5, 10, 15):")
        bot.register_next_step_handler(msg, step_ask_m)

    def step_ask_m(msg):
        chat_id = msg.chat.id
        if chat_id not in user_flows: return
        try: marks = int(msg.text.strip())
        except ValueError: marks = 5
        q = user_flows[chat_id]['q']
        prof = get_profile(chat_id)

        ans = assistant.generate_syllabus_notes(q, marks=marks, subject=prof['subject'], institution=prof['institution'], level=prof['level'])
        bot.send_message(chat_id, ans)
        del user_flows[chat_id]

    print("Bot polling initialized.")
    bot.infinity_polling()
