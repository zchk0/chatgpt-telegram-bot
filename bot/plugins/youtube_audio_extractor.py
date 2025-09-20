import logging
import re
from typing import Dict

from yt_dlp import YoutubeDL, DownloadError
import imageio_ffmpeg as iio_ffmpeg

from .plugin import Plugin

class YouTubeAudioExtractorPlugin(Plugin):
    def get_source_name(self) -> str:
        return "YouTube Audio Extractor"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "extract_youtube_audio",
            "description": "Extract audio (only MP3 320 kbps) from a YouTube video",
            "parameters": {
                "type": "object",
                "properties": {
                    "youtube_link": {"type": "string", "description": "YouTube video link to extract audio from"}
                },
                "required": ["youtube_link"],
            },
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        link = kwargs["youtube_link"]
        # Получаем (и при необходимости подкачаем) бинарник ffmpeg
        try:
            ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            logging.warning(f"Failed to obtain ffmpeg binary via imageio-ffmpeg: {e}")
            ffmpeg_path = None  # yt-dlp попробует сам без постпроцессинга

        # Пробуем заранее узнать название для безопасного имени файла
        title = "audio"
        try:
            with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(link, download=False)
                title = info.get("title") or title
        except Exception as e:
            logging.warning(f"Failed to fetch video info: {e}")

        safe_title = re.sub(r"[^\w\-_\. ]", "_", title)
        outtmpl = f"{safe_title}.%(ext)s"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }

        # Если ffmpeg есть — сконвертируем в mp3
        if ffmpeg_path:
            ydl_opts["ffmpeg_location"] = ffmpeg_path
            ydl_opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320",
            }]
            output_path = f"{safe_title}.mp3"
        else:
            # Фолбэк: скачиваем без конвертации (скорее всего .m4a/.webm)
            # Это убирает зависимость от ffmpeg, но формат не mp3.
            ydl_opts["postprocessors"] = []
            output_path = None  # узнаем после скачивания

        try:
            with YoutubeDL(ydl_opts) as ydl:
                res = ydl.download([link])

            # Если не конвертировали — определим фактическое имя файла
            if not ffmpeg_path:
                # yt-dlp кладёт итоговый файл по шаблону outtmpl, расширение выбирается по формату
                # Можно повторно дернуть extract_info с download=False и взять ydl.prepare_filename(info),
                # но мы уже скачали, поэтому делаем ещё один безопасный вызов:
                with YoutubeDL({"quiet": True, "no_warnings": True}) as ydl2:
                    info2 = ydl2.extract_info(link, download=False)
                    guessed = ydl2.prepare_filename(info2)
                # Файл уже скачан, guessed укажет на исходное расширение, напр. .m4a / .webm
                output_path = re.sub(r"\.webm$|\.m4a$|\.mp4$", lambda m: m.group(0), guessed)

            return {
                "direct_result": {
                    "kind": "file",
                    "format": "path",
                    "value": output_path
                }
            }
        except DownloadError as e:
            logging.warning(f"yt-dlp download error: {str(e)}")
            return {"result": "Failed to extract audio"}
        except Exception as e:
            logging.warning(f'Failed to extract audio from YouTube video: {str(e)}')
            return {'result': 'Failed to extract audio'}
