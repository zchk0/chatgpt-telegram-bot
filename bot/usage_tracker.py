import os.path
import pathlib
import json
from datetime import date


def year_month(date_str):
    # extract string of year-month from date, eg: '2023-03'
    return str(date_str)[:7]


def _to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


class UsageTracker:
    """
    UsageTracker class
    Enables tracking of daily/monthly usage per user.
    User files are stored as JSON in /usage_logs directory.
    JSON example:
    {
        "user_name": "@user_name",
        "user_timezone": "Europe/London",
        "current_cost": {
            "day": 0.45,
            "month": 3.23,
            "all_time": 3.23,
            "last_update": "2023-03-14"},
        "usage_history": {
            "chat_tokens": {
                "2023-03-13": 520,
                "2023-03-14": 1532
            },
            "transcription_seconds": {
                "2023-03-13": 125,
                "2023-03-14": 64
            },
            "number_images": {
                "2023-03-12": [0, 2, 3],
                "2023-03-13": [1, 2, 3],
                "2023-03-14": [0, 1, 2]
            }
        }
    }
    """

    def __init__(self, user_id, user_name, logs_dir="usage_logs"):
        """
        Initializes UsageTracker for a user with current date.
        Loads usage data from usage log file.
        :param user_id: Telegram ID of the user
        :param user_name: Telegram user name
        :param logs_dir: path to directory of usage logs, defaults to "usage_logs"
        """
        self.user_id = user_id
        self.logs_dir = logs_dir
        # path to usage file of given user
        self.user_file = f"{logs_dir}/{user_id}.json"

        if os.path.isfile(self.user_file):
            with open(self.user_file, "r") as file:
                self.usage = json.load(file)
            # гарантируем наличие новых секций
            uh = self.usage.setdefault("usage_history", {})
            uh.setdefault("vision_tokens", {})
            uh.setdefault("tts_characters", {})
            uh.setdefault("number_images", {})
            uh.setdefault("chat_tokens", {})
            uh.setdefault("transcription_seconds", {})
        else:
            # ensure directory exists
            pathlib.Path(logs_dir).mkdir(exist_ok=True)
            # create new dictionary for this user
            self.usage = {
                "user_name": user_name,
                "user_timezone": None,
                "current_cost": {
                    "day": 0.0,
                    "month": 0.0,
                    "all_time": 0.0,
                    "last_update": str(date.today()),
                },
                "usage_history": {
                    "chat_tokens": {},
                    "transcription_seconds": {},
                    "number_images": {},            # по дням: [cnt_256, cnt_512, cnt_1024]
                    "tts_characters": {},           # по моделям: {'tts-1': {date: chars}, ...}
                    "vision_tokens": {},            # по дням: tokens
                },
            }


    def set_user_timezone(self, timezone=None):
        """
        Add time zone for current user
        """
        self.usage["user_timezone"] = timezone
        with open(self.user_file, "w") as outfile:
            json.dump(self.usage, outfile)

    def get_user_timezone(self):
        """
        Get time zone for current user
        """
        return self.usage.get('user_timezone', None)

    # token usage functions:

    def add_chat_tokens(self, tokens, tokens_price=0.002):
        """Adds used tokens from a request to a users usage history and updates current cost
        :param tokens: total tokens used in last request
        :param tokens_price: price per 1000 tokens, defaults to 0.002
        """
        tokens = _to_int(tokens, 0)
        today_key = str(date.today())
        token_cost = round(tokens * _to_float(tokens_price, 0.0) / 1000, 6)
        self.add_current_costs(token_cost)

        # update usage_history
        hist = self.usage["usage_history"]["chat_tokens"]
        hist[today_key] = _to_int(hist.get(today_key, 0), 0) + tokens

        # write updated token usage to user file
        with open(self.user_file, "w") as outfile:
            json.dump(self.usage, outfile)

    def get_current_token_usage(self):
        """Get token amounts used for today and this month

        :return: total number of tokens used per day and per month
        """
        today_key = str(date.today())
        usage_day = _to_int(self.usage["usage_history"]["chat_tokens"].get(today_key, 0), 0)
        month = today_key[:7]  # year-month as string
        usage_month = 0
        for d, t in self.usage["usage_history"]["chat_tokens"].items():
            if d.startswith(month):
                usage_month += _to_int(t, 0)
        return usage_day, usage_month

    # image usage functions:

    def add_image_request(self, image_size, image_prices="0.016,0.018,0.02"):
        """Add image request to users usage history and update current costs.

        :param image_size: requested image size
        :param image_prices: prices for images of sizes ["256x256", "512x512", "1024x1024"],
                             defaults to [0.016, 0.018, 0.02]
        """
        sizes = ["256x256", "512x512", "1024x1024"]
        requested_size = sizes.index(image_size)

        if isinstance(image_prices, str):
            image_prices_list = [float(x) for x in image_prices.split(",")]
        else:
            image_prices_list = [float(x) for x in image_prices]

        image_cost = _to_float(image_prices_list[requested_size], 0.0)
        self.add_current_costs(image_cost)

        # update usage_history
        today_key = str(date.today())
        hist = self.usage["usage_history"]["number_images"]
        if today_key not in hist:
            hist[today_key] = [0, 0, 0]
        hist[today_key][requested_size] = _to_int(hist[today_key][requested_size], 0) + 1

        # write updated image number to user file
        with open(self.user_file, "w") as outfile:
            json.dump(self.usage, outfile)

    def get_current_image_count(self):
        """Get number of images requested for today and this month.

        :return: total number of images requested per day and per month
        """
        today_key = str(date.today())
        usage_day = sum(_to_int(x, 0) for x in self.usage["usage_history"]["number_images"].get(today_key, [0, 0, 0]))
        month = today_key[:7]
        usage_month = 0
        for d, arr in self.usage["usage_history"]["number_images"].items():
            if d.startswith(month):
                usage_month += sum(_to_int(x, 0) for x in arr)
        return usage_day, usage_month


    # vision usage functions
    def add_vision_tokens(self, tokens, vision_token_price=0.01):
        """
         Adds requested vision tokens to a users usage history and updates current cost.
        :param tokens: total tokens used in last request
        :param vision_token_price: price per 1K tokens transcription, defaults to 0.01
        """
        tokens = _to_int(tokens, 0)
        today_key = str(date.today())

        token_price = round(tokens * _to_float(vision_token_price, 0.0) / 1000, 2)
        self.add_current_costs(token_price)

        # update usage_history
        hist = self.usage["usage_history"]["vision_tokens"]
        hist[today_key] = _to_int(hist.get(today_key, 0), 0) + tokens

        # write updated token usage to user file
        with open(self.user_file, "w") as outfile:
            json.dump(self.usage, outfile)

    def get_current_vision_tokens(self):
        """Get vision tokens for today and this month.

        :return: total amount of vision tokens per day and per month
        """
        today_key = str(date.today())
        tokens_day = _to_int(self.usage["usage_history"]["vision_tokens"].get(today_key, 0), 0)
        month = today_key[:7]
        tokens_month = 0
        for d, t in self.usage["usage_history"]["vision_tokens"].items():
            if d.startswith(month):
                tokens_month += _to_int(t, 0)
        return tokens_day, tokens_month

    # tts usage functions:

    def add_tts_request(self, text_length, tts_model, tts_prices):
        """
        tts_model: 'tts-1' | 'tts-1-hd'
        tts_prices: либо строка "0.015,0.030", либо список [0.015, 0.030] (цена за 1K символов)
        """
        text_length = _to_int(text_length, 0)

        models = ["tts-1", "tts-1-hd"]
        idx = models.index(tts_model)

        if isinstance(tts_prices, str):
            tts_prices_list = [float(x) for x in tts_prices.split(",")]
        else:
            tts_prices_list = [float(x) for x in tts_prices]

        price_per_k = _to_float(tts_prices_list[idx], 0.0)
        tts_price = round(text_length * price_per_k / 1000, 2)
        self.add_current_costs(tts_price)

        today_key = str(date.today())
        hist_all = self.usage["usage_history"].setdefault("tts_characters", {})
        hist_model = hist_all.setdefault(tts_model, {})
        hist_model[today_key] = _to_int(hist_model.get(today_key, 0), 0) + text_length

        # write updated token usage to user file
        with open(self.user_file, "w") as outfile:
            json.dump(self.usage, outfile)

    def get_current_tts_usage(self):
        """Get length of speech generated for today and this month.

        :return: total amount of characters converted to speech per day and per month
        """
        models = ["tts-1", "tts-1-hd"]
        today_key = str(date.today())

        characters_day = 0
        for m in models:
            if m in self.usage["usage_history"]["tts_characters"] and \
               today_key in self.usage["usage_history"]["tts_characters"][m]:
                characters_day += _to_int(self.usage["usage_history"]["tts_characters"][m][today_key], 0)

        month = today_key[:7]
        characters_month = 0
        for m in models:
            for d, cnt in self.usage["usage_history"]["tts_characters"].get(m, {}).items():
                if d.startswith(month):
                    characters_month += _to_int(cnt, 0)

        return int(characters_day), int(characters_month)


    # transcription usage functions:

    def add_transcription_seconds(self, seconds, minute_price=0.006):
        """Adds requested transcription seconds to a users usage history and updates current cost.
        :param seconds: total seconds used in last request
        :param minute_price: price per minute transcription, defaults to 0.006
        """
        seconds = _to_int(seconds, 0)
        today_key = str(date.today())

        transcription_price = round(seconds * _to_float(minute_price, 0.0) / 60, 2)
        self.add_current_costs(transcription_price)

        # update usage_history
        hist = self.usage["usage_history"]["transcription_seconds"]
        hist[today_key] = _to_int(hist.get(today_key, 0), 0) + seconds

        # write updated token usage to user file
        with open(self.user_file, "w") as outfile:
            json.dump(self.usage, outfile)

    def add_current_costs(self, request_cost):
        """
        Add current cost to all_time, day and month cost and update last_update date.
        """
        request_cost = _to_float(request_cost, 0.0)
        today = date.today()
        last_update = date.fromisoformat(self.usage["current_cost"]["last_update"])

        # add to all_time cost, initialize with calculation of total_cost if key doesn't exist
        self.usage["current_cost"]["all_time"] = _to_float(
            self.usage["current_cost"].get("all_time", self.initialize_all_time_cost()), 0.0
        ) + request_cost
        # add current cost, update new day
        if today == last_update:
            self.usage["current_cost"]["day"] = _to_float(self.usage["current_cost"]["day"], 0.0) + request_cost
            self.usage["current_cost"]["month"] = _to_float(self.usage["current_cost"]["month"], 0.0) + request_cost
        else:
            if today.month == last_update.month:
                self.usage["current_cost"]["month"] = _to_float(self.usage["current_cost"]["month"], 0.0) + request_cost
            else:
                self.usage["current_cost"]["month"] = request_cost
            self.usage["current_cost"]["day"] = request_cost
            self.usage["current_cost"]["last_update"] = str(today)

    def get_current_transcription_duration(self):
        """Get minutes and seconds of audio transcribed for today and this month.

        :return: total amount of time transcribed per day and per month (4 values)
        """
        today_key = str(date.today())
        seconds_day = _to_int(self.usage["usage_history"]["transcription_seconds"].get(today_key, 0), 0)
        month = today_key[:7]
        seconds_month = 0
        for d, s in self.usage["usage_history"]["transcription_seconds"].items():
            if d.startswith(month):
                seconds_month += _to_int(s, 0)
        minutes_day, seconds_day = divmod(seconds_day, 60)
        minutes_month, seconds_month = divmod(seconds_month, 60)
        return int(minutes_day), round(seconds_day, 2), int(minutes_month), round(seconds_month, 2)

    # general functions
    def get_current_cost(self):
        """Get total USD amount of all requests of the current day and month

        :return: cost of current day and month
        """
        today = date.today()
        last_update = date.fromisoformat(self.usage["current_cost"]["last_update"])
        if today == last_update:
            cost_day = _to_float(self.usage["current_cost"]["day"], 0.0)
            cost_month = _to_float(self.usage["current_cost"]["month"], 0.0)
        else:
            cost_day = 0.0
            cost_month = _to_float(self.usage["current_cost"]["month"], 0.0) if today.month == last_update.month else 0.0

        # add to all_time cost, initialize with calculation of total_cost if key doesn't exist
        cost_all_time = _to_float(
            self.usage["current_cost"].get("all_time", self.initialize_all_time_cost()), 0.0
        )
        return {"cost_today": cost_day, "cost_month": cost_month, "cost_all_time": cost_all_time}

    def initialize_all_time_cost(self, tokens_price=0.002, image_prices="0.016,0.018,0.02", minute_price=0.006, vision_token_price=0.01, tts_prices='0.015,0.030'):
        """Get total USD amount of all requests in history
        
        :param tokens_price: price per 1000 tokens, defaults to 0.002
        :param image_prices: prices for images of sizes ["256x256", "512x512", "1024x1024"],
            defaults to [0.016, 0.018, 0.02]
        :param minute_price: price per minute transcription, defaults to 0.006
        :param vision_token_price: price per 1K vision token interpretation, defaults to 0.01
        :param tts_prices: price per 1K characters tts per model ['tts-1', 'tts-1-hd'], defaults to [0.015, 0.030]
        :return: total cost of all requests
        """

        # chat tokens
        total_tokens = sum(_to_int(v, 0) for v in self.usage["usage_history"]["chat_tokens"].values())
        token_cost = round(total_tokens * _to_float(tokens_price, 0.0) / 1000, 6)

        # images
        image_prices_list = [float(x) for x in (image_prices.split(",") if isinstance(image_prices, str) else image_prices)]
        ni = list(self.usage["usage_history"]["number_images"].values())
        if ni:
            cols = list(zip(*ni))  # три столбца по размерам
            total_images = [sum(_to_int(v, 0) for v in col) for col in cols]
        else:
            total_images = [0, 0, 0]
        image_cost = sum(cnt * price for cnt, price in zip(total_images, image_prices_list))

        # transcription
        total_transcription_seconds = sum(_to_int(v, 0) for v in self.usage["usage_history"]["transcription_seconds"].values())
        transcription_cost = round(total_transcription_seconds * _to_float(minute_price, 0.0) / 60, 2)

        # vision
        total_vision_tokens = sum(_to_int(v, 0) for v in self.usage["usage_history"]["vision_tokens"].values())
        vision_cost = round(total_vision_tokens * _to_float(vision_token_price, 0.0) / 1000, 2)

        # tts
        tts_prices_list = [float(x) for x in (tts_prices.split(",") if isinstance(tts_prices, str) else tts_prices)]
        if self.usage["usage_history"]["tts_characters"]:
            total_characters = [sum(_to_int(v, 0) for v in model.values())
                                for model in self.usage["usage_history"]["tts_characters"].values()]
        else:
            total_characters = [0] * len(tts_prices_list)
        tts_cost = round(sum(cnt * price / 1000 for cnt, price in zip(total_characters, tts_prices_list)), 2)

        return token_cost + transcription_cost + image_cost + vision_cost + tts_cost
