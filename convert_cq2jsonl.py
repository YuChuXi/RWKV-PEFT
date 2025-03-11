import re
import os
import tqdm
import time
import threading
from dataclasses import dataclass


@dataclass
class CQObject:
    type: str
    params: dict


@dataclass
class Message:
    username: str
    qq: str
    group: str
    content: list


def cq_code_unescape(cq_code):
    """
    将CQ码中的HTML实体编码反转义为原始字符。
    """
    # 定义HTML实体编码与原始字符的映射关系
    escape_map = {
        "&#91;": "[",
        "&#93;": "]",
        "&#44;": ",",
        "&amp;": "&",
    }

    # 遍历映射关系，替换HTML实体编码为原始字符
    for escaped, original in escape_map.items():
        cq_code = cq_code.replace(escaped, original)

    return cq_code


def parse_content(content):
    parts = []
    cq_pattern = re.compile(r"\[CQ:(.*?)\]", re.DOTALL)
    start = 0
    for match in cq_pattern.finditer(content):
        text_part = content[start : match.start()]
        if text_part:
            parts.append(text_part)
        cq_str = match.group(1)
        cq_parts = cq_str.split(",", 1)
        cq_type = cq_parts[0]
        params_str = cq_parts[1] if len(cq_parts) > 1 else ""
        params = {}
        if params_str:
            params_str = cq_code_unescape(params_str)
            temp_marker = "\x00"
            param_str_processed = params_str.replace("&#44;", temp_marker)
            param_list = param_str_processed.split(",")
            for param in param_list:
                param = param.replace(temp_marker, ",")
                if "=" in param:
                    key, value = param.split("=", 1)
                    key = cq_code_unescape(key.strip())
                    value = cq_code_unescape(value.strip())
                    params[key] = value
        parts.append(CQObject(cq_type, params))
        start = match.end()
    text_part = content[start:]
    if text_part:
        parts.append(text_part)
    return parts


def parse_messages(text):
    entries = re.split(r"(\n\n<\|[^>]+?\|>: )", text)
    messages = []
    header_pattern = re.compile(r"<\|([^@]+?)@(\d+)(?:\(([^)]+)\))?\|>:")
    for i in range(1, len(entries), 2):
        header_str = entries[i].strip()
        content = entries[i + 1].lstrip("\n").rstrip("\n")
        match = header_pattern.match(header_str)
        if not match:
            continue
        username = match.group(1)
        qq = match.group(2)
        group = match.group(3) or ""
        content_parts = parse_content(content)
        messages.append(Message(username, qq, group, content_parts))
    return messages


def process_file(file_set, file_set_lock, user_set, user_set_lock):
    local_user_set = 0
    while True:
        with file_set_lock:
            if len(file_set) == 0:
                return
            file_path = file_set.pop()

        with open(file_path, "r", encoding="utf-8") as f:
            input_text = f.read()

        messages = parse_messages(input_text)

        local_user_set = len(messages)
        
        with user_set_lock:
            user_set.append(local_user_set)
            local_user_set = 0


if __name__ == "__main__":
    log_folder = "data/message_cache_25_03_10_18_22/"

    user_set = []
    user_set_lock = threading.Lock()

    file_set = set(map(lambda x: log_folder + x, os.listdir(log_folder)))
    file_set_lock = threading.Lock()

    for i in range(4):
        threading.Thread(
            target=process_file, args=(file_set, file_set_lock, user_set, user_set_lock)
        ).start()

    while True:
        time.sleep(1)
        print(len(file_set), sum(user_set))
