import re
import os
import tqdm
import time
import numpy
import joblib
import random
import concurrent.futures
from typing import Generator

from dataclasses import dataclass
from typing import Dict, List, Set

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from rwkvt.dataset.binidx import MMapIndexedDataset

tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

CTX = 8192


@dataclass()
class CQObject:
    __slots__ = ["type", "params"]
    type: str
    params: Dict[str, str]


@dataclass()
class Message:
    __slots__ = ["username", "qq", "group", "content"]
    username: str
    qq: str
    group: str
    content: list


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=numpy.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
    def add_item(self, np_array):
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)
    def end_document(self):
        self._doc_idx.append(len(self._sizes))
    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
    

class LOG_Processer:
    def __init__(
        self,
    ):
        self.log_folder = "data/message_cache_25_03_10_18_22/"
        self.obj_folder = "data/message_cache_obj/"
        self.log_output = "data/message_cache"

        self.group_set: Dict[str, Message] = {}

        self.entries_pattern = re.compile(r"(\n\n<\|[^>]+?\|>: )")
        self.header_pattern = re.compile(r"<\|([^@]+?)@(\d+)(?:\(([^)]+)\))?\|>:")
        self.cq_pattern = re.compile(r"\[CQ:(.*?)\]", re.DOTALL)
        self.escape_list = [
            ("&#91;", "["),
            ("&#93;", "]"),
            ("&#44;", ","),
            ("&amp;", "&"),
        ]

        self.user_blocklist = []
        self.cq_whitelist_1 = ["at", "face", "increase", "mface"] # 会被基本原样保留的CQ码
        self.cq_whitelist_2 = self.cq_whitelist_1 + ["image", "video", "file"] # 会被保留type的CQ码

    def cq_code_unescape(self, cq_code):
        """
        将CQ码中的HTML实体编码反转义为原始字符。
        """
        for (escaped, original) in self.escape_list:
            cq_code = cq_code.replace(escaped, original)

        return cq_code
    
    def cq_code_escape(self, cq_code):
        """
        将CQ码中的HTML实体编码反转义为原始字符。
        """
        for (escaped, original) in self.escape_list[::-1]:
            cq_code = cq_code.replace(original, escaped)

        return cq_code

    def parse_content(self, content):
        parts = []
        start = 0
        for match in self.cq_pattern.finditer(content):
            text_part = content[start : match.start()]
            if text_part:
                parts.append(text_part)
            cq_str = match.group(1)
            cq_parts = cq_str.split(",", 1)
            cq_type = cq_parts[0]
            params_str = cq_parts[1] if len(cq_parts) > 1 else ""
            params = {}
            if params_str:
                params_str = self.cq_code_unescape(params_str)
                temp_marker = "\x00"
                param_str_processed = params_str.replace("&#44;", temp_marker)
                param_list = param_str_processed.split(",")
                for param in param_list:
                    param = param.replace(temp_marker, ",")
                    if "=" in param:
                        key, value = param.split("=", 1)
                        key = self.cq_code_unescape(key.strip())
                        value = self.cq_code_unescape(value.strip())
                        params[key] = value
            parts.append(CQObject(cq_type, params))
            start = match.end()
        text_part = content[start:]
        if text_part:
            parts.append(text_part)
        return parts

    def parse_messages(self, text):
        entries = self.entries_pattern.split(text)
        messages = []
        for i in tqdm.trange(1, len(entries), 2, leave=False, position=0):
            header_str = entries[i].strip()
            content = entries[i + 1].lstrip("\n").rstrip("\n")
            match = self.header_pattern.match(header_str)
            if not match:
                continue
            username = match.group(1)
            qq = match.group(2)
            group = match.group(3) or ""
            content_parts = self.parse_content(content)
            message = Message(username, qq, group, content_parts)
            if message is not None:
                messages.append(message)
        return messages

    def log_to_obj(self, log_file, obj_file):
        with open(log_file, "r", encoding="utf-8") as f:
            input_text = f.read()
        obj = self.parse_messages(input_text)
        joblib.dump(obj, obj_file)
        return obj

    def load_obj(self, name):
        try:
            return joblib.load(self.obj_folder + name + ".gz")
        except:
            print(f"Load {name} from log")
            return self.log_to_obj(
                self.log_folder + name, self.obj_folder + name + ".gz"
            )

    def wash_message(self, message: Message):
        if message.qq in self.user_blocklist:
            return None
        message.content = [
            cqobj
            for cqobj in message.content
            if (not isinstance(cqobj, CQObject)) or cqobj.type in self.cq_whitelist_2
        ]
        if len(message.content) == 0:
            return None
        return message

    def message_to_text(self, message: Message) -> str:
        """將Message對象轉換為處理後的文本"""
        parts = []
        for part in message.content:
            if isinstance(part, str):
                parts.append(part.replace("\n", "\\n"))
            elif isinstance(part, CQObject):
                if part.type in self.cq_whitelist_1:
                    cq_str = f"[CQ:{part.type}"
                    for k, v in part.params.items():
                        cq_str += f",{k}={self.cq_code_escape(v)}"
                    cq_str += "]"
                    parts.append(cq_str)
                else:
                    parts.append(f"[CQ:{part.type}]")
        return "".join(parts)

    def process_group(self, group_name: str, messages: List[Message], n_tokens: int = 0) -> List[List[int]]:
        """處理單個群組的消息並切片"""
        slices = []
        current_tokens = []
        pad_token = 65530
        end_token = 65531
        
        for msg in messages:
            text = self.message_to_text(msg)
            msg_tokens = tokenizer.encode(text)
            
            if n_tokens > 0 and current_tokens:
                current_tokens += [pad_token] * (n_tokens-1) + [end_token]
            
            # 添加當前消息token
            current_tokens.extend(msg_tokens)
            
            # 切片處理
            while len(current_tokens) >= CTX:
                slice_tokens = current_tokens[:CTX]
                slices.append(slice_tokens)
                current_tokens = current_tokens[CTX:]
        
        return slices

    def shuffle_and_export(self, all_slices: Set[List[int]], output_path: str):
        """混洗数据并导出为binidx格式"""
        # 使用集合进行初步去重和混洗
        
        # 创建数据集构建器
        builder = MMapIndexedDatasetBuilder(output_path, dtype=numpy.uint16)
        
        # 添加所有切片
        for tokens in tqdm.tqdm(all_slices, desc="Exporting"):
            print(len(tokens))
            builder.add_item(numpy.array(tokens, dtype=numpy.uint16))
            
        # 添加特殊结束标记
        builder.end_document()
        builder.finalize()

    def convert_all(self):
        os.makedirs(self.obj_folder, exist_ok=True)
        
        # 流式处理并立即写入
        output_path = self.log_output
        builder = MMapIndexedDatasetBuilder(output_path, dtype=numpy.uint16)
        
        # 使用生成器管道处理数据
        def process_pipeline():
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                file_list = os.listdir(self.log_folder)
                for future in concurrent.futures.as_completed(
                    [executor.submit(self.load_obj, f) for f in file_list]
                ):
                    group_name, messages = future.result()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as group_exec:
                        for slice_future in concurrent.futures.as_completed(
                            [group_exec.submit(self.process_group, group_name, messages)]
                        ):
                            yield from slice_future.result()
        
        # 分批写入
        buffer = []
        buffer_size = 1000  # 根据内存调整批处理大小
        for tokens in tqdm.tqdm(process_pipeline(), desc="Processing"):
            buffer.append(tokens)
            if len(buffer) >= buffer_size:
                random.shuffle(buffer)
                for item in buffer:
                    builder.add_item(numpy.array(item, dtype=numpy.uint16))
                buffer.clear()

        # 写入剩余数据
        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                builder.add_item(numpy.array(item, dtype=numpy.uint16))
        
        builder.end_document()
        builder.finalize()

LOG_Processer().convert_all()
