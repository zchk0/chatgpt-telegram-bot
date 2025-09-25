from __future__ import annotations
import copy
import datetime
import logging
import os
import uuid

import tiktoken

import openai

import requests
import json
import httpx
import io
from PIL import Image

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from utils import is_direct_result, encode_image, decode_image
from plugin_manager import PluginManager

# Models can be found here: https://platform.openai.com/docs/models/overview
# Models gpt-3.5-turbo-0613 and  gpt-3.5-turbo-16k-0613 will be deprecated on June 13, 2024
GPT_3_MODELS = ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613")
GPT_3_16K_MODELS = ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125")
GPT_4_MODELS = ("gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-turbo-preview")
GPT_4_32K_MODELS = ("gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613")
# Vision-модели прошлых поколений
GPT_4_VISION_MODELS = ("gpt-4o",)
GPT_4_128K_MODELS = ("gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4-turbo-2024-04-09")
GPT_4O_MODELS = ("gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest")
O_MODELS = ("o1", "o1-mini", "o1-preview")
# GPT-5 (мультимодальные; большой общий контекст)
GPT_5_MODELS = ("gpt-5",)
GPT_5_MINI_MODELS = ("gpt-5-mini",)
GPT_5_NANO_MODELS = ("gpt-5-nano",)
GPT_5_ALL_MODELS = GPT_5_MODELS + GPT_5_MINI_MODELS + GPT_5_NANO_MODELS

# Полный список
GPT_ALL_MODELS = (
    GPT_3_MODELS
    + GPT_3_16K_MODELS
    + GPT_4_MODELS
    + GPT_4_32K_MODELS
    + GPT_4_VISION_MODELS
    + GPT_4_128K_MODELS
    + GPT_4O_MODELS
    + GPT_5_ALL_MODELS
    + O_MODELS
)

# Семейства, требующие ключа `max_completion_tokens` и механизма tools/tool_calls
REASONING_MODELS = O_MODELS + GPT_5_ALL_MODELS


def default_max_tokens(model: str) -> int:
    """
    Gets the default number of max tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    base = 1200
    if model in GPT_3_MODELS:
        return base
    elif model in GPT_4_MODELS:
        return base * 2
    elif model in GPT_3_16K_MODELS:
        if model == "gpt-3.5-turbo-1106":
            return 4096
        return base * 4
    elif model in GPT_4_32K_MODELS:
        return base * 8
    elif model in GPT_4_VISION_MODELS:
        return 4096
    elif model in GPT_4_128K_MODELS:
        return 4096
    elif model in GPT_4O_MODELS:
        return 4096
    elif model in O_MODELS:
        return 4096
    elif model in GPT_5_ALL_MODELS:
        return 4096


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    if model in ("gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-32k-0314", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"):
        return False
    if model in O_MODELS:
        return False
    if model in GPT_5_ALL_MODELS:
        # Старое API функций недоступно для reasoning-моделей
        return False
    return True


# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    """
    Return translated text for a key in specified bot_language.
    Keys and translations can be found in the translations.json.
    """
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        # Fallback to English if the translation is not available
        if key in translations.get('en', {}):
            return translations['en'][key]
        logging.warning(f"No english definition found for key '{key}' in translations.json")
        # return key as text
        return key


class OpenAIHelper:
    """
    ChatGPT helper class.
    """

    def __init__(self, config: dict, plugin_manager: PluginManager):
        """
        Initializes the OpenAI helper class with the given configuration.
        :param config: A dictionary containing the GPT configuration
        :param plugin_manager: The plugin manager
        """
        http_client = httpx.AsyncClient(proxy=config['proxy']) if 'proxy' in config else None
        self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client)
        self.config = config
        self.plugin_manager = plugin_manager
        self.conversations: dict[int: list] = {}          # {chat_id: history}
        self.conversations_vision: dict[int: bool] = {}   # {chat_id: is_vision}
        self.last_updated: dict[int: datetime] = {}       # {chat_id: last_update_timestamp}
        self.current_telegram_chat_id = 0
        self.current_telegram_user_id = 0
        self.current_telegram_user_name = ""
        self.usage_tracker = {}

    def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.__count_tokens(self.conversations[chat_id])

    def __extract_openai_error_message(self, e: Exception) -> str:
        """
        Хелпер для извлечения ошибок и безопасного для телеграмм форматирования
        """
        try:
            body = getattr(e, "response", None)
            if body is not None:
                try:
                    j = body.json()
                except Exception:
                    j = None
                if j and isinstance(j, dict):
                    # Вернём message; подробности (param/code) используем на вызывающей стороне
                    return j.get("error", {}).get("message", str(e))
        except Exception:
            pass
        return str(e)

    def __telegram_safe(self, text: str) -> str:
        # Урезаем спецсимволы MarkdownV2 — чтобы Telegram не падал
        bad = "*_[]()~`>|#=+-{}.!\\"
        for ch in bad:
            text = text.replace(ch, " ")
        return text

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, str]:
        """
        Gets a full response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        plugins_used = ()

        # Авто-выход из vision при первом текстовом запросе
        await self.__auto_exit_vision_if_needed(chat_id)

        response = await self.__common_get_chat_response(chat_id, query)
        if (self.config['enable_functions'] or self.config.get('enable_tools', True)):
            response, plugins_used = await self.__handle_function_or_tool_call(chat_id, response)
            if is_direct_result(response):
                return response, '0'

        answer = ''
        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = choice.message.content.strip()
                if index == 0:
                    self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n{content}\n\n'
        else:
            answer = response.choices[0].message.content.strip()
            self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config['bot_language']
        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        safe_total_tokens = self._safe_total_tokens(response, self.__count_tokens(self.conversations[chat_id]))
        if self.config['show_usage']:
            answer += (
                "\n\n---\n"
                f"💰 {str(safe_total_tokens)} {localized_text('stats_tokens', bot_language)}"
                f" ({str(response.usage.prompt_tokens)} {localized_text('prompt', bot_language)},"
                f" {str(response.usage.completion_tokens)} {localized_text('completion', bot_language)})"
            )
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, safe_total_tokens

    async def get_chat_response_stream(self, chat_id: int, query: str, params: dict):
        """
        Stream response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :param params: Additional parameters as a dictionary
        :return: Yields (partial_text, 'not_finished') during stream and (final_text, tokens_used) at the end.
        """
        self.current_telegram_chat_id = chat_id
        self.current_telegram_user_id = params.get('telegram_user_id', 0)
        self.current_telegram_user_name = params.get('telegram_user_name', None)
        self.usage_tracker = params.get('usage_tracker', {})
        plugins_used = ()

        # сброс истории при отсуствии chat_id
        if chat_id not in self.conversations or chat_id not in self.conversations_vision or self.__max_age_reached(chat_id):
            self.reset_chat_history(chat_id)
        
        # Авто-выход из vision при первом текстовом запросе
        await self.__auto_exit_vision_if_needed(chat_id)

        # Определяем модель и режим стрима
        effective_model = self.config['model'] if not self.conversations_vision.get(chat_id, False) else self.config['vision_model']
        want_stream = params.get("want_stream", True)
        if effective_model in REASONING_MODELS:
            want_stream = False

        try:
            response = await self.__common_get_chat_response(chat_id, query, stream=want_stream)
            streaming = want_stream
        except Exception as e:
            em = str(e).lower()
            if ("verified to stream this model" in em) or ("param" in em and "stream" in em and "unsupported_value" in em):
                response = await self.__common_get_chat_response(chat_id, query, stream=False)
                streaming = False
            else:
                raise

        tools_enabled = (self.config.get('enable_functions') or self.config.get('enable_tools', True))
        if streaming:
            answer = ""
            added_to_history = False
            # Аккумуляторы для tool_calls / function_call (старый стиль)
            tool_calls_acc = []
            function_call_acc = None
            saw_tools = False

            async for chunk in response:
                if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                    continue
                ch = chunk.choices[0]
                delta = getattr(ch, "delta", None)

                # Обычные текстовые дельты — отдаём пользователю сразу
                if delta is not None and getattr(delta, "content", None):
                    if not saw_tools:
                        answer += delta.content
                        yield answer, 'not_finished'

                # Новый стиль tools: delta.tool_calls
                if delta is not None and getattr(delta, "tool_calls", None):
                    saw_tools = True
                    for tc in delta.tool_calls:
                        # расширяем список под индекс
                        while len(tool_calls_acc) <= tc.index:
                            tool_calls_acc.append({"id": None, "function": {"name": "", "arguments": ""}})
                        if getattr(tc, "id", None):
                            tool_calls_acc[tc.index]["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn:
                            if getattr(fn, "name", None):
                                tool_calls_acc[tc.index]["function"]["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                tool_calls_acc[tc.index]["function"]["arguments"] += fn.arguments

                # Старый стиль функций: delta.function_call
                if delta is not None and getattr(delta, "function_call", None):
                    saw_tools = True
                    if getattr(delta.function_call, "name", None):
                        function_call_acc = function_call_acc or {"name": "", "arguments": ""}
                        function_call_acc["name"] += delta.function_call.name
                    if getattr(delta.function_call, "arguments", None):
                        function_call_acc = function_call_acc or {"name": "", "arguments": ""}
                        function_call_acc["arguments"] += delta.function_call.arguments

                # Сигнал завершения шага: tool_calls / function_call → пора вызывать инструменты
                finish = getattr(ch, "finish_reason", None)
                if tools_enabled and finish in ("tool_calls", "function_call"):
                    break

            # Если инструментов не было — обычный стрим без tools
            if not (tools_enabled and (tool_calls_acc or function_call_acc)):
                final_text = answer.strip()
                if final_text:
                    self.__add_to_history(chat_id, role="assistant", content=final_text)
                    added_to_history = True

                tokens_used = str(self.__count_tokens(self.conversations[chat_id]))
                # show_usage / плагины (если были из предыдущих шагов — маловероятно в этой ветке)
                show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
                plugin_names = tuple(set(self.plugin_manager.get_plugin_source_name(p) for p in plugins_used))
                if self.config['show_usage']:
                    final_text += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
                    if show_plugins_used:
                        final_text += f"\n🔌 {', '.join(plugin_names)}"
                elif show_plugins_used:
                    final_text += f"\n\n---\n🔌 {', '.join(plugin_names)}"

                yield final_text, tokens_used
                return

            # ==== Ветка с tools ====
            # Делаем обработку инструментов НЕстримово (чтобы заполнить историю правильными сообщениями tool/assistant)
            # __handle_function_or_tool_call сама дочитает необходимые данные из response, добавит tool ответы и при необходимости
            # подготовит историю для follow-up запроса.
            resp_after_tools, plugins_used = await self.__handle_function_or_tool_call(
                chat_id, response, stream=False, times=0, plugins_used=()
            )
            if is_direct_result(resp_after_tools):
                # Прямой результат плагина (файл, фото, dice ...)
                yield resp_after_tools, '0'
                return

            # Follow-up запрос после инструментов — уже стримим финальный текст
            m = self.config['model'] if not self.conversations_vision.get(chat_id, False) else self.config['vision_model']
            max_key = 'max_completion_tokens' if m in REASONING_MODELS else 'max_tokens'
            followup = await self.client.chat.completions.create(
                model=m,
                messages=self.conversations[chat_id],
                **{max_key: self.config['max_tokens']},
                stream=True
            )

            followup_answer = ""
            async for ch2 in followup:
                if not getattr(ch2, "choices", None) or len(ch2.choices) == 0:
                    continue
                d = getattr(ch2.choices[0], "delta", None)
                if d is not None and getattr(d, "content", None):
                    followup_answer += d.content
                    # Стримим уже «пост-инструментальный» текст; до этого могли быть прелюдии в answer
                    out = (answer + followup_answer) if answer else followup_answer
                    yield out, 'not_finished'

            final_text = ((answer + followup_answer) if answer else followup_answer).strip()
            if final_text:
                self.__add_to_history(chat_id, role="assistant", content=final_text)

            tokens_used = str(self.__count_tokens(self.conversations[chat_id]))
            show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
            plugin_names = tuple(set(self.plugin_manager.get_plugin_source_name(p) for p in plugins_used))
            if self.config['show_usage']:
                final_text += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
                if show_plugins_used:
                    final_text += f"\n🔌 {', '.join(plugin_names)}"
            elif show_plugins_used:
                final_text += f"\n\n---\n🔌 {', '.join(plugin_names)}"

            yield final_text, tokens_used
            return

        if tools_enabled:
            response, plugins_used = await self.__handle_function_or_tool_call(chat_id, response, stream=False)
            if is_direct_result(response):
                yield response, '0'
                return

        # Сборка полноразмерного ответа без стрима
        answer = ''
        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = (choice.message.content or "").strip()
                if index == 0:
                    self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n{content}\n\n'
        else:
            answer = (response.choices[0].message.content or "").strip()
            self.__add_to_history(chat_id, role="assistant", content=answer)

        tokens_used = str(self.__count_tokens(self.conversations[chat_id]))

        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(set(self.plugin_manager.get_plugin_source_name(p) for p in plugins_used))
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        yield answer, tokens_used

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            self.__add_to_history(chat_id, role="user", content=query)

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['max_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.__add_to_history(chat_id, role="user", content=query)
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            effective_model = self.config['model'] if not self.conversations_vision[chat_id] else self.config['vision_model']
            max_tokens_key = 'max_completion_tokens' if effective_model in REASONING_MODELS else 'max_tokens'

            common_args = {
                'model': effective_model,
                'messages': self.conversations[chat_id],
                'temperature': self.config['temperature'],
                'n': self.config['n_choices'],
                max_tokens_key: self.config['max_tokens'],
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream
            }

            # Подключаем инструменты:
            function_specs = self.plugin_manager.get_functions_specs()
            has_specs = len(function_specs) > 0
            if has_specs:
                if effective_model in REASONING_MODELS:
                    common_args['tools'] = [{"type": "function", "function": f} for f in function_specs]
                    common_args['tool_choice'] = 'auto'
                else:
                    common_args['functions'] = function_specs
                    common_args['function_call'] = 'auto'

            return await self.client.chat.completions.create(**common_args)

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            msg = self.__telegram_safe(self.__extract_openai_error_message(e))
            raise Exception(f"⚠️ {localized_text('openai_invalid', bot_language)}. ⚠️\n{msg}") from e

        except Exception as e:
            msg = self.__telegram_safe(str(e))
            raise Exception(f"⚠️ {localized_text('error', bot_language)}. ⚠️\n{msg}") from e

    async def __handle_function_or_tool_call(self, chat_id, response, stream=False, times=0, plugins_used=()):
        """
        Унифицированная обработка вызовов инструментов:
        - Новый стиль (GPT-5/О): assistant.tool_calls -> tool-ответы -> догоняющий запрос
        - Старый стиль (классические модели): function_call -> function (role=function) -> догоняющий запрос
        """
        import json

        def _is_reasoning() -> bool:
            effective_model = self.config['model'] if not self.conversations_vision[chat_id] else self.config['vision_model']
            return effective_model in REASONING_MODELS

        def _safe_json_args(raw: str) -> str:
            """
            Возвращает строку JSON для передачи в plugin; если raw невалидный JSON, оборачиваем как {"_raw": raw}
            """
            try:
                json.loads(raw if raw else "{}")
                return raw if raw else "{}"
            except Exception:
                return json.dumps({"_raw": raw})

        # Извлекаем tool_calls (новый стиль) и/или function_call (старый стиль) из response ----
        tool_calls = []
        function_call = None

        if stream:
            # Стрим: нужно собрать кусочки tool_calls / function_call
            async for item in response:
                if len(item.choices) == 0:
                    continue
                first = item.choices[0]

                # Новый стиль (tool_calls)
                if getattr(first.delta, "tool_calls", None):
                    for tc in first.delta.tool_calls:
                        # гарантируем длину массива
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({"id": None, "function": {"name": "", "arguments": ""}})
                        # накапливаем id и поля функции
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls[tc.index]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

                # Старый стиль (function_call)
                if getattr(first.delta, "function_call", None):
                    if first.delta.function_call.name:
                        function_call = function_call or {"name": "", "arguments": ""}
                        function_call["name"] += first.delta.function_call.name
                    if first.delta.function_call.arguments:
                        function_call = function_call or {"name": "", "arguments": ""}
                        function_call["arguments"] += first.delta.function_call.arguments

                # финализация
                if getattr(first, "finish_reason", None) in ("tool_calls", "function_call", "stop"):
                    break

            # если вообще никаких вызовов — возвращаем исходный response наверх
            if (not tool_calls) and (function_call is None):
                return response, plugins_used

        else:
            # Non-stream: всё приходит готовым в единственном сообщении
            if len(response.choices) == 0:
                return response, plugins_used
            msg = response.choices[0].message

            # Новый стиль
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "function": {
                            "name": (tc.function.name if tc.function else "") or "",
                            "arguments": (tc.function.arguments if (tc.function and tc.function.arguments) else "") or ""
                        }
                    })

            # Старый стиль
            if getattr(msg, "function_call", None):
                function_call = {
                    "name": msg.function_call.name or "",
                    "arguments": msg.function_call.arguments or ""
                }

            if (not tool_calls) and (function_call is None):
                return response, plugins_used

        # Обработка НОВОГО стиля tools/tool_calls
        if tool_calls:
            # Зафиксируем сообщение ассистента с tool_calls (это обязательное предшествующее сообщение)
            normalized_calls = self.__add_assistant_with_tool_calls(chat_id, tool_calls)

            # На каждый tool_call — вызов реального плагина и добавление role=tool с тем же tool_call_id
            for tc in normalized_calls:
                fname = tc["function"]["name"]
                args_raw = tc["function"]["arguments"] or ""
                args_json_str = _safe_json_args(args_raw)

                logging.info(f'Calling tool {fname} with arguments {args_json_str}')
                tool_result = await self.plugin_manager.call_function(fname, self, args_json_str)

                if fname not in plugins_used:
                    plugins_used += (fname,)

                if is_direct_result(tool_result):
                    # Добавим stub в историю и вернём результат наверх (бот уже отправил ответ пользователю)
                    self.__add_tool_result_to_history(
                        chat_id, tool_call_id=tc["id"],
                        content=json.dumps({'result': 'Done, the content has been sent to the user.'})
                    )
                    return tool_result, plugins_used

                # Ответ инструмента
                self.__add_tool_result_to_history(chat_id, tool_call_id=tc["id"], content=tool_result)

            # Делаем догоняющий запрос: теперь у модели есть tool_calls + ответы tool
            m = self.config['model'] if not self.conversations_vision[chat_id] else self.config['vision_model']
            max_key = 'max_completion_tokens' if m in REASONING_MODELS else 'max_tokens'
            response = await self.client.chat.completions.create(
                model=m,
                messages=self.conversations[chat_id],
                tools=[{"type": "function", "function": f} for f in self.plugin_manager.get_functions_specs()],
                tool_choice='auto' if times < self.config['functions_max_consecutive_calls'] else 'none',
                **{max_key: self.config['max_tokens']},
                stream=stream
            )
            # Рекурсивно продолжаем обработку до завершения цепочки инструментов
            return await self.__handle_function_or_tool_call(chat_id, response, stream, times + 1, plugins_used)

        # Обработка СТАРОГО стиля functions/function_call
        if function_call is not None:
            fname = function_call["name"]
            args_json_str = _safe_json_args(function_call["arguments"] or "")

            logging.info(f'Calling function {fname} with arguments {args_json_str}')
            fn_result = await self.plugin_manager.call_function(fname, self, args_json_str)

            if fname not in plugins_used:
                plugins_used += (fname,)

            if is_direct_result(fn_result):
                # совместимость со старым стилем: пишем role=function
                self.__add_function_call_to_history(
                    chat_id=chat_id, function_name=fname,
                    content=json.dumps({'result': 'Done, the content has been sent to the user.'})
                )
                return fn_result, plugins_used

            # Записываем ответ функции в историю (старый стиль)
            self.__add_function_call_to_history(chat_id=chat_id, function_name=fname, content=fn_result)

            # Делаем догоняющий запрос
            m = self.config['model'] if not self.conversations_vision[chat_id] else self.config['vision_model']
            max_key = 'max_completion_tokens' if m in REASONING_MODELS else 'max_tokens'
            response = await self.client.chat.completions.create(
                model=m,
                messages=self.conversations[chat_id],
                functions=self.plugin_manager.get_functions_specs(),
                function_call='auto' if times < self.config['functions_max_consecutive_calls'] else 'none',
                **{max_key: self.config['max_tokens']},
                stream=stream
            )
            return await self.__handle_function_or_tool_call(chat_id, response, stream, times + 1, plugins_used)

        # Если сюда дошли — ничего вызывать не нужно
        return response, plugins_used


    def __add_assistant_with_tool_calls(self, chat_id: int, tool_calls: list[dict]) -> list[dict]:
        """
        Добавляет в историю сообщение ассистента с полем tool_calls (новый API).
        Возвращает нормализованный список tool_calls (c гарантированными id).
        """
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)

        normalized = []
        for tc in tool_calls:
            # ожидаем структуру {"id": ..., "function": {"name": ..., "arguments": "..."}}
            tid = tc.get("id") or f"toolcall_{uuid.uuid4().hex}"
            fn = tc.get("function", {}) or {}
            normalized.append({
                "id": tid,
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or ""
                }
            })
        # важно: само сообщение ассистента ДОЛЖНО содержать tool_calls
        self.conversations[chat_id].append({
            "role": "assistant",
            "content": "",           # обычно пусто
            "tool_calls": normalized
        })
        return normalized

    def __add_tool_result_to_history(self, chat_id: int, tool_call_id: str, content: str):
        """
        Сообщение с ролью 'tool' — ответ инструмента на конкретный tool_call.
        """
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        self.conversations[chat_id].append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

    def __add_function_call_to_history(self, chat_id, function_name, content):
        """
        Старый стиль (для совместимости со старыми моделями).
        """
        self.conversations[chat_id].append({"role": "function", "name": function_name, "content": content})

    async def generate_image(self, prompt: str) -> tuple[str, str]:
        bot_language = self.config['bot_language']
        try:
            response = await self.client.images.generate(
                prompt=prompt,
                n=1,
                model=self.config['image_model'],
                quality=self.config['image_quality'],
                style=self.config['image_style'],
                size=self.config['image_size']
            )

            if len(response.data) == 0:
                logging.error(f'No response from GPT: {str(response)}')
                raise Exception(
                    f"⚠️ {localized_text('error', bot_language)}. "
                    f"⚠️\n{localized_text('try_again', bot_language)}."
                )

            return response.data[0].url, self.config['image_size']
        except Exception as e:
            raise Exception(f"⚠️ {localized_text('error', bot_language)}. ⚠️\n{self.__telegram_safe(str(e))}") from e

    async def generate_speech(self, text: str) -> tuple[any, int]:
        """
        Generates an audio from the given text using TTS model.
        :param prompt: The text to send to the model
        :return: The audio in bytes and the text size
        """
        bot_language = self.config['bot_language']
        try:
            response = await self.client.audio.speech.create(
                model=self.config['tts_model'],
                voice=self.config['tts_voice'],
                input=text,
                response_format='opus'
            )
            temp_file = io.BytesIO()
            temp_file.write(response.read())
            temp_file.seek(0)
            return temp_file, len(text)
        except Exception as e:
            raise Exception(f"⚠️ {localized_text('error', bot_language)}. ⚠️\n{self.__telegram_safe(str(e))}") from e

    async def transcribe(self, filename):
        """
        Transcribes the audio file using the Whisper model.
        """
        try:
            with open(filename, "rb") as audio:
                prompt_text = self.config['whisper_prompt']
                result = await self.client.audio.transcriptions.create(model="whisper-1", file=audio, prompt=prompt_text)
                return result.text
        except Exception as e:
            logging.exception(e)
            raise Exception(f"⚠️ {localized_text('error', self.config['bot_language'])}. ⚠️\n{self.__telegram_safe(str(e))}") from e

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response_vision(self, chat_id: int, content: list, stream=False):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            # сохраняем в историю «для нас»
            if self.config.get('enable_vision_follow_up_questions', False):
                self.conversations_vision[chat_id] = True
                self.__add_to_history(chat_id, role="user", content=content)
            else:
                # чтобы не раздувать историю, кладём только текст
                query = ""
                for message in content:
                    if message.get("type") in ("text", "input_text"):
                        query = message.get("text", "")
                        break
                self.__add_to_history(chat_id, role="user", content=query)

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['vision_max_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'[VISION] Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    last = self.conversations[chat_id][-1]
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.conversations[chat_id] += [last]
                except Exception as e:
                    logging.warning(f'[VISION] Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            message = {'role': 'user', 'content': content}

            effective_model = self.config['vision_model']
            max_tokens_key = 'max_completion_tokens' if effective_model in REASONING_MODELS else 'max_tokens'
            common_args = {
                'model': effective_model,
                'messages': self.conversations[chat_id][:-1] + [message],
                'temperature': self.config['temperature'],
                'n': 1,
                max_tokens_key: self.config['vision_max_tokens'],
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream
            }
            if stream:
                common_args['stream_options'] = {"include_usage": True}

            # логируем что отправляем для распознавания
            log_args = copy.deepcopy(common_args)
            for m in log_args.get("messages", []):
                if isinstance(m.get("content"), list):
                    for c in m["content"]:
                        if c.get("type") == "image_url":
                            url = c["image_url"].get("url", "")
                            if len(url) > 50:
                                c["image_url"]["url"] = url[:50] + "..."
            logging.info(f"[VISION] Sending request: {log_args}")

            # логируем что получаем после распознавания
            resp = await self.client.chat.completions.create(**common_args)
            logging.info(f"[VISION] Received response: {resp}")
            return resp

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            msg = self.__telegram_safe(self.__extract_openai_error_message(e))
            raise Exception(f"⚠️ {localized_text('openai_invalid', bot_language)}. ⚠️\n{msg}") from e

        except Exception as e:
            msg = self.__telegram_safe(str(e))
            raise Exception(f"⚠️ {localized_text('error', bot_language)}. ⚠️\n{msg}") from e


    async def interpret_image(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        def _to_data_url(b64: str, mime: str = "image/png") -> str:
            # encode_image(fileobj) может вернуть «голый» base64 — превратим в data URL
            if not b64:
                return ""
            if b64.startswith("data:"):
                return b64
            return f"data:{mime};base64,{b64}"

        def _extract_text_from_choice(choice) -> str:
            """
            Универсальный экстрактор текста:
            - message.output_text
            - message.content: str | list[parts{type: text|output_text|refusal|reasoning, text/...}]
            - message.refusal / message.reasoning (строки)
            """
            msg = getattr(choice, "message", None)
            if msg is None:
                return ""

            try:
                ot = getattr(msg, "output_text", None)
                if ot:
                    t = (ot or "").strip()
                    if t:
                        return t
            except Exception:
                pass

            c = getattr(msg, "content", "")
            if isinstance(c, str):
                t = c.strip()
                if t:
                    return t

            parts_text = []

            if isinstance(c, list):
                for p in c:
                    # dict-части
                    if isinstance(p, dict):
                        typ = p.get("type")
                        # допустимые типы, в которых может лежать человекочитаемый текст
                        if typ in {"text", "output_text", "refusal", "reasoning"}:
                            # разные модели кладут текст в разных ключах
                            t = p.get("text") or p.get("output_text") or p.get("refusal") or p.get("reasoning") or ""
                            if t:
                                parts_text.append(t)
                    else:
                        # объектные части
                        try:
                            typ = getattr(p, "type", "")
                            if typ in ("text", "output_text", "refusal", "reasoning"):
                                t = (
                                    getattr(p, "text", "")
                                    or getattr(p, "output_text", "")
                                    or getattr(p, "refusal", "")
                                    or getattr(p, "reasoning", "")
                                    or ""
                                )
                                if t:
                                    parts_text.append(t)
                        except Exception:
                            pass

            if parts_text:
                return "".join(parts_text).strip()

            # Иногда SDK кладёт отказ/рассуждения в отдельные поля
            try:
                ref = getattr(msg, "refusal", None)
                if ref and isinstance(ref, str) and ref.strip():
                    return ref.strip()
            except Exception:
                pass

            try:
                rsn = getattr(msg, "reasoning", None)
                if rsn and isinstance(rsn, str) and rsn.strip():
                    return rsn.strip()
            except Exception:
                pass

            # Если пришли tool_calls — текста может не быть
            if getattr(msg, "tool_calls", None):
                logging.info("[VISION] tool_calls present; content empty")
                return ""

            return (str(c) if c is not None else "").strip()

        async def _ask_with_content(content_parts):
            resp = await self.__common_get_chat_response_vision(chat_id, content_parts, stream=False)
            choices = getattr(resp, "choices", []) or []
            ans = ""
            if choices:
                text = _extract_text_from_choice(choices[0])
                ans = (text or "").strip()
                if ans:
                    self.__add_to_history(chat_id, role="assistant", content=ans)

            # токены
            tokens_used = None
            try:
                if getattr(resp, "usage", None) is not None:
                    tokens_used = self._safe_total_tokens(resp, self.__count_tokens(self.conversations[chat_id]))
            except Exception:
                tokens_used = None
            if tokens_used is None:
                tokens_used = self.__count_tokens(self.conversations[chat_id])

            # полезные логи
            try:
                fr = getattr(choices[0], "finish_reason", None) if choices else None
                ct = type(getattr(choices[0].message, "content", None)).__name__ if choices and getattr(choices[0], "message", None) else None
                logging.info(f"[VISION] finish_reason={fr}, content_type={ct}, extracted_len={len(ans)}")
            except Exception:
                pass

            return ans, str(tokens_used), resp

        # подготовка контента
        b64 = encode_image(fileobj)
        data_url = _to_data_url(b64, mime="image/png")
        prompt_text = self.config['vision_prompt'] if prompt is None else prompt

        content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url, "detail": self.config.get("vision_detail", "auto")}},
        ]
        answer, tokens_used, resp = await _ask_with_content(content)

        if not answer:
            answer = "Извините, модель вернула пустой ответ. Попробуйте ещё раз или пришлите фото крупнее/светлее. Если проблема повторится увеличьте vision_max_tokens"

        # show_usage
        if self.config.get("show_usage"):
            bot_language = self.config["bot_language"]
            answer += (
                "\n\n---\n"
                f"💰 {tokens_used} {localized_text('stats_tokens', bot_language)}"
            )
            try:
                if getattr(resp, "usage", None) is not None:
                    answer += (
                        f" ({str(resp.usage.prompt_tokens)} {localized_text('prompt', bot_language)}, "
                        f"{str(resp.usage.completion_tokens)} {localized_text('completion', bot_language)})"
                    )
            except Exception:
                pass

        return answer, tokens_used


    async def interpret_image_stream(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type': 'text', 'text': prompt}, {'type': 'image_url',
                    'image_url': {'url': image, 'detail': self.config['vision_detail']}}]

        want_stream = self.config.get("vision_want_stream", True)
        if self.config['vision_model'] in REASONING_MODELS:
            want_stream = False

        try:
            response = await self.__common_get_chat_response_vision(chat_id, content, stream=want_stream)
        except Exception as e:
            em = str(e).lower()
            if ("verified to stream this model" in em) or ("param" in em and "stream" in em and "unsupported_value" in em):
                response = await self.__common_get_chat_response_vision(chat_id, content, stream=False)
                want_stream = False
            else:
                raise

        answer = ''
        if want_stream:
            async for chunk in response:
                if len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    answer += delta.content
                    yield answer, 'not_finished'
            answer = answer.strip()
            self.__add_to_history(chat_id, role="assistant", content=answer)
            tokens_used = str(self.__count_tokens(self.conversations[chat_id]))
            if self.config['show_usage']:
                answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
            yield answer, tokens_used
        else:
            if len(response.choices) > 1 and self.config['n_choices'] > 1:
                for index, choice in enumerate(response.choices):
                    content = choice.message.content.strip()
                    if index == 0:
                        self.__add_to_history(chat_id, role="assistant", content=content)
                    answer += f'{index + 1}\u20e3\n{content}\n\n'
            else:
                answer = response.choices[0].message.content.strip()
                self.__add_to_history(chat_id, role="assistant", content=answer)
            tokens_used = str(self.__count_tokens(self.conversations[chat_id]))
            if self.config['show_usage']:
                answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
            yield answer, tokens_used

    def reset_chat_history(self, chat_id, content=''):
        """
        Resets the conversation history.
        """
        if content == '':
            content = self.config['assistant_prompt']
        self.conversations[chat_id] = [{"role": "assistant" if self.config['model'] in O_MODELS else "system", "content": content}]
        self.conversations_vision[chat_id] = False

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config['max_conversation_age_minutes']
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    def __add_to_history(self, chat_id, role, content):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: 'system' | 'user' | 'assistant' | 'tool' | 'function'
        :param content: str | list (для vision-сообщений)
        """
        # на случай, если история ещё не инициализирована
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        self.conversations[chat_id].append({"role": role, "content": content})

    async def __summarise(self, conversation, max_tokens: int = 256) -> str:
        """
        Summarises the conversation history.
        :param conversation: list[dict] | any — часть истории
        :param max_tokens: лимит токенов для сводки
        """
        import json as _json
        prompt = (
            "Summarize the prior exchange (possibly about an image) in a compact way. "
            "Keep user intent and key facts; drop chit-chat. "
            f"Target <= {max_tokens} tokens."
        )
        messages = [
            {"role": "assistant", "content": prompt},
            {"role": "user", "content": _json.dumps(conversation, ensure_ascii=False)}
        ]
        m = self.config['model']
        max_key = 'max_completion_tokens' if m in REASONING_MODELS else 'max_tokens'
        response = await self.client.chat.completions.create(
            model=m,
            messages=messages,
            **{max_key: max(64, int(max_tokens))}
        )
        return (response.choices[0].message.content or "").strip()

    def __max_model_tokens(self):
        base = 4096
        if self.config['model'] in GPT_3_MODELS:
            return base
        if self.config['model'] in GPT_3_16K_MODELS:
            return base * 4
        if self.config['model'] in GPT_4_MODELS:
            return base * 2
        if self.config['model'] in GPT_4_32K_MODELS:
            return base * 8
        if self.config['model'] in GPT_4_VISION_MODELS:
            return base * 31
        if self.config['model'] in GPT_4_128K_MODELS:
            return base * 31
        if self.config['model'] in GPT_4O_MODELS:
            return base * 31
        if self.config['model'] in GPT_5_ALL_MODELS:
            return 128_000
        elif self.config['model'] in O_MODELS:
            # https://platform.openai.com/docs/models#o1
            if self.config['model'] == "o1":
                return 100_000
            elif self.config['model'] == "o1-preview":
                return 32_768
            else:
                return 65_536
        raise NotImplementedError(
            f"Max tokens for model {self.config['model']} is not implemented yet."
        )

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def __count_tokens(self, messages) -> int:
        """
        Counts the number of tokens required to send the given messages.
        :param messages: the messages to send
        :return: the number of tokens required
        """
        model = self.config['model']
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # GPT-5 использует семейство o200k; пробуем Harmony → Base, затем cl100k
            try:
                if model in GPT_5_ALL_MODELS:
                    try:
                        encoding = tiktoken.get_encoding("o200k_harmony")
                    except Exception:
                        encoding = tiktoken.get_encoding("o200k_base")
                else:
                    encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")

        if model in GPT_ALL_MODELS:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == 'content':
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    else:
                        for message1 in value:
                            if message1['type'] == 'image_url':
                                image = decode_image(message1['image_url']['url'])
                                num_tokens += self.__count_tokens_vision(image)
                            else:
                                num_tokens += len(encoding.encode(message1['text']))
                else:
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    else:
                        num_tokens += len(encoding.encode(str(value)))
                    if key == "name":
                        num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def __count_tokens_vision(self, image_bytes: bytes) -> int:
        """
        Counts the number of tokens for interpreting an image.
        :param image_bytes: image to interpret
        :return: the number of tokens required
        """
        image_file = io.BytesIO(image_bytes)
        image = Image.open(image_file)
        model = self.config['vision_model']
        if model not in (GPT_4_VISION_MODELS + GPT_5_ALL_MODELS):
            raise NotImplementedError(f"""count_tokens_vision() is not implemented for model {model}.""")

        w, h = image.size
        if w > h:
            w, h = h, w
        base_tokens = 85
        detail = self.config['vision_detail']
        if detail == 'low':
            return base_tokens
        elif detail in ('high', 'auto'): # assuming worst cost for auto
            f = max(w / 768, h / 2048)
            if f > 1:
                w, h = int(w / f), int(h / f)
            tw, th = (w + 511) // 512, (h + 511) // 512
            tiles = tw * th
            num_tokens = base_tokens + tiles * 170
            return num_tokens
        else:
            raise NotImplementedError(f"""unknown parameter detail={detail} for model {model}.""")

    def get_current_telegram_chat_user_info(self) -> int:
        """
        Get current telegram chat and user info
        """
        return {
            "chat_id": self.current_telegram_chat_id,
            "user_id": self.current_telegram_user_id,
            "user_name": self.current_telegram_user_name,
            "usage_tracker": self.usage_tracker
        }
    
    def _safe_total_tokens(self, resp, fallback: int) -> int:
        """
        Безопасно извлекаем usage tokens
        """
        try:
            u = getattr(resp, "usage", None)
            if u is not None and getattr(u, "total_tokens", None) is not None:
                return int(u.total_tokens)
        except Exception:
            pass
        return int(fallback)


    def _extract_text_from_choice(self, choice) -> str:
        """
        Универсально достаёт текст из ответа (Chat Completions/мультимодаль).
        Покрывает варианты: content=str, content=list[parts], output_text, tool_calls.
        """
        # Иногда SDK кладёт итог в message.output_text
        try:
            ot = getattr(choice.message, "output_text", None)
            if ot:
                return (ot or "").strip()
        except Exception:
            pass

        # Стандартное поле
        msg = getattr(choice, "message", None)
        if msg is None:
            return ""

        content = getattr(msg, "content", "")
        # content=string
        if isinstance(content, str):
            t = content.strip()
            if t:
                return t

        # content=list частей (мультимодаль/«reasoning»)
        parts_text = []
        if isinstance(content, list):
            for p in content:
                # dict-части нового формата
                if isinstance(p, dict):
                    t = p.get("text") or p.get("output_text") or ""
                    if p.get("type") in {"text", "output_text"} and t:
                        parts_text.append(t)
                else:
                    # объект с атрибутами
                    try:
                        typ = getattr(p, "type", "")
                        if typ in ("text", "output_text"):
                            t = getattr(p, "text", "") or getattr(p, "output_text", "") or ""
                            if t:
                                parts_text.append(t)
                    except Exception:
                        pass
            if parts_text:
                return "".join(parts_text).strip()

        # Если контент пуст, посмотрим tool-calls (часто тогда content="")
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # Лог для диагностики и простой текст на всякий случай
            try:
                logging.info("[VISION] tool_calls present: %s", repr(tool_calls)[:400])
            except Exception:
                pass
            # Если не выполняете инструменты, лучше не пытаться «склеивать» их аргументы в ответ.
            # Вернём пусто — пусть сработает ваш fallback.
            return ""

        # fallback
        return (str(content) if content is not None else "").strip()


    async def __auto_exit_vision_if_needed(self, chat_id: int):
        """
        Если включён exit_vision_on_text и чат сейчас в vision-режиме,
        сворачиваем историю с изображением в короткую сводку и выходим из vision.
        Конфиг берём из self.config: строки 'true'/'false', числа — как строки.
        """
        exit_on_text = self.config.get('exit_vision_on_text', False)
        keep_last_n = int(self.config.get('vision_exit_keep_last_n', '0'))
        sum_tokens  = int(self.config.get('vision_exit_summary_tokens', '256'))

        logging.info(f"Text after a photo vision. exit_on_text={exit_on_text}, "
             f"is_vision={self.conversations_vision.get(chat_id)}, "
             f"keep_last_n={keep_last_n}, summary_tokens={sum_tokens}")
        
        if not exit_on_text:
            return
        if not self.conversations_vision.get(chat_id, False):
            return
        if chat_id not in self.conversations or not self.conversations[chat_id]:
            return

        history = self.conversations[chat_id]
        head = history[0]                  # system / первый элемент
        body = history[1:]                 # остальная история (в т.ч. image-сообщения)

        # Делаем сводку body с лимитом токенов
        try:
            summary = await self.__summarise(body, max_tokens=sum_tokens)
        except Exception:
            summary = "Не удалось сгенерировать сводку предыдущего обсуждения изображения."

        # Берём «сырой» хвост, если нужно (напр., последний ответ ассистента)
        if keep_last_n > 0:
            raw_tail = body[-keep_last_n:]
            tail = []
            for m in raw_tail:
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    tail.append(m)
                # всё, что не строка (списки/картинки/файлы/tool и т.п.) — отбрасываем
        else:
            tail = []

        # Пересобираем историю: system + краткая сводка + опционный хвост
        new_history = [
            head,
            {"role": "assistant", "content": f"Краткая сводка обсуждения изображения: {summary}"}
        ]
        new_history.extend(tail)
        self.conversations[chat_id] = new_history
        # Выходим из vision-режима
        self.conversations_vision[chat_id] = False
