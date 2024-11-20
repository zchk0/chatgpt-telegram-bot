import os
import datetime
import json
import asyncio
from typing import Dict, Union
from uuid import uuid4
import logging
from telegram import Bot
from telegram.error import TelegramError

from .plugin import Plugin

# Author: https://github.com/zchk0
class Reminder(Plugin):
    """
    A plugin to create, manage, and delete single or recurring reminders with file persistence
    """

    def __init__(self):
        self.reminders_file = 'reminders.json'
        self.reminders = self.load_reminders()
        self.checking_task = None
        self.bot = Bot(token=os.environ.get('TELEGRAM_BOT_TOKEN', ''))

    def get_source_name(self) -> str:
        return "Reminder by zchk0"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "get_current_time",
            "description": "Get the current time before creating and editing reminders",
        }, {
            "name": "add_reminder",
            "description": "Create a single or recurring reminder",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {"type": "string", "description": "A simple, human-readable set of characters (up to 8) for a unique reminder ID to be told to the user"},
                    "message": {"type": "string", "description": "Reminder message"},
                    "datetime": {"type": "string", "description": "ISO 8601 format date and time for the reminder"},
                    "repeat": {
                        "type": "string",
                        "description": "Frequency of recurrence: 'daily', 'weekly', 'monthly', or 'none'",
                        "enum": ["none", "daily", "weekly", "monthly"]
                    }
                },
                "required": ["reminder_id", "message", "datetime"]
            },
        }, {
            "name": "add_multiple_reminders",
            "description": "Adding multiple reminders in one request",
        }, {
            "name": "edit_reminder",
            "description": "Edit a message or reminder date by its ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {"type": "string", "description": "ID of the reminder"},
                    "message": {"type": "string", "description": "New reminder message"},
                    "datetime": {"type": "string", "description": "ISO 8601 format date and time for the reminder"},
                },
                "required": ["reminder_id"]
            },
        }, {
            "name": "remove_reminder",
            "description": "Remove a reminder by its ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "reminder_id": {"type": "string", "description": "ID of the reminder to remove"}
                },
                "required": ["reminder_id"]
            },
        }, {
            "name": "remove_reminders_for_current_chat",
            "description": "Delete all reminders for the current chat",
        }, {
            "name": "start_reminders_task",
            "description": "Enable reminder checking",
        }]

    def get_current_time(self):
        return datetime.datetime.now(datetime.timezone.utc).astimezone()

    def load_reminders(self):
        """
        Load reminders from a JSON file.
        """
        if os.path.exists(self.reminders_file):
            with open(self.reminders_file, 'r') as f:
                return json.load(f)
        return {}

    def save_reminders(self):
        """
        Save reminders to a JSON file.
        """
        with open(self.reminders_file, 'w') as f:
            json.dump(self.reminders, f)

    async def add_reminder(self, reminder_id: str, chat_id: int, message: str, datetime_str: str, repeat: str):
        try:
            remind_time = datetime.datetime.fromisoformat(datetime_str)
            # Если временная зона отсутствует, добавляем временную зону сервера
            if remind_time.tzinfo is None:
                server_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
                remind_time = remind_time.replace(tzinfo=server_timezone)
        except ValueError:
            logging.info('Invalid date format')
            return {"error": "Invalid date format"}

        # if reminder_id in self.reminders:
        #     logging.info(f"Reminder with ID {reminder_id} already exists")
        #     return {"error": f"Reminder with ID {reminder_id} already exists"}

        # Сохранение напоминания с датой, включающей временную зону
        self.reminders[reminder_id] = {
            "chat_id": chat_id,
            "message": message,
            "time": remind_time.isoformat(),
            "repeat": repeat
        }
        self.save_reminders()
        self.start_reminders_task()

        return {
            "reminder_id": reminder_id,
            "message": message,
            "time": remind_time.isoformat(),
            "repeat": repeat
        }

    def add_multiple_reminders(self):
        return {"error": "You cannot add multiple reminders at once."}

    def start_reminders_task(self):
        """
        Starts the reminder checking task if it is not already running.
        """
        self.reminders = self.load_reminders()
        if not self.checking_task or self.checking_task.done():
            self.checking_task = asyncio.create_task(self.check_and_send_reminders())
            logging.info("Reminder checking task started.")
        else:
            logging.info("Reminder checking task is already running.")

    def remove_reminder(self, reminder_id: str):
        if reminder_id in self.reminders:
            self.reminders.pop(reminder_id)
            self.save_reminders()
            return {"message": "Reminder removed successfully", "reminder_id": reminder_id}
        else:
            return {"error": "Reminder ID not found"}

    def remove_reminders_for_current_chat(self, chat_id):
        to_remove = [r_id for r_id, reminder in self.reminders.items() if reminder["chat_id"] == chat_id]
        for r_id in to_remove:
            self.reminders.pop(r_id)
        self.save_reminders()
        return {"message": f"All reminders removed successfully for current chat"}

    def edit_reminder(self, reminder_id: str, message: str = None, datetime_str: str = None):
        self.reminders = self.load_reminders()
        if reminder_id not in self.reminders:
            logging.info(f"Reminder with ID {reminder_id} does not exist")
            return {"error": f"Reminder with ID {reminder_id} does not exist"}
        
        reminder = self.reminders[reminder_id]

        # Обновление времени напоминания, если указано новое значение
        if datetime_str:
            try:
                remind_time = datetime.datetime.fromisoformat(datetime_str)
                if remind_time.tzinfo is None:
                    server_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
                    remind_time = remind_time.replace(tzinfo=server_timezone)
                reminder["time"] = remind_time.isoformat()
            except ValueError:
                logging.info('Invalid date format')
                return {"error": "Invalid date format"}

        # Обновление сообщения, если указано новое значение
        if message:
            reminder["message"] = message

        self.reminders[reminder_id] = reminder
        self.save_reminders()

        return {
            "reminder_id": reminder_id,
            "message": reminder["message"],
            "time": reminder["time"],
            "repeat": reminder.get("repeat", "none")
        }

    async def check_and_send_reminders(self):
        while True:
            now = datetime.datetime.now(datetime.timezone.utc)
            to_remove = []
            for reminder_id, reminder in self.reminders.items():
                remind_time = datetime.datetime.fromisoformat(reminder["time"]).astimezone()
                chat_id = reminder["chat_id"]
                message = reminder["message"]

                if now >= remind_time:
                    logging.info(f'send_remind {now}')
                    try:
                        await self.bot.send_message(
                            chat_id=chat_id,
                            text=f"{message} \n\n🔑 `{reminder_id}`",
                            parse_mode='Markdown'
                        )
                    except TelegramError as e:
                        logging.error(f"Failed to send message: {e}")
                    
                    if reminder["repeat"] == 'none':
                        to_remove.append(reminder_id)
                    elif reminder["repeat"] == 'daily':
                        self.reminders[reminder_id]["time"] = (remind_time + datetime.timedelta(days=1)).isoformat()
                    elif reminder["repeat"] == 'weekly':
                        self.reminders[reminder_id]["time"] = (remind_time + datetime.timedelta(weeks=1)).isoformat()
                    elif reminder["repeat"] == 'monthly':
                        next_month = (remind_time.month % 12) + 1
                        year = remind_time.year + (1 if next_month == 1 else 0)
                        self.reminders[reminder_id]["time"] = remind_time.replace(year=year, month=next_month).isoformat()

            for reminder_id in to_remove:
                self.reminders.pop(reminder_id)
            self.save_reminders()

            await asyncio.sleep(60)

    async def execute(self, function_name, helper, **kwargs) -> dict:
        if function_name == 'add_reminder':
            reminder_id = kwargs.get('reminder_id', str(uuid4()))
            message = kwargs.get('message', '')
            datetime_str = kwargs.get('datetime', '')
            repeat = kwargs.get('repeat', 'none')
            return await self.add_reminder(reminder_id, helper.get_current_telegram_chat_id(), message, datetime_str, repeat)

        elif function_name == 'remove_reminder':
            reminder_id = kwargs.get('reminder_id', '')
            return self.remove_reminder(reminder_id)

        elif function_name == 'remove_reminders_for_current_chat':
            return self.remove_reminders_for_current_chat(helper.get_current_telegram_chat_id())

        elif function_name == 'start_reminders_task':
            return self.start_reminders_task()
        
        elif function_name == 'add_multiple_reminders':
            return self.add_multiple_reminders()
        
        elif function_name == 'get_current_time':
            return self.get_current_time()

        elif function_name == 'edit_reminder':
            reminder_id = kwargs.get('reminder_id', '')
            message = kwargs.get('message', None)
            datetime_str = kwargs.get('datetime', None)
            return self.edit_reminder(reminder_id, message, datetime_str)

        return {"error": "Unknown error"}
