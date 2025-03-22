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
from typing import Dict, List, Set, Tuple

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from rwkvt.dataset.binidx import MMapIndexedDataset

tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

CTX = 4096


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
        
        if os.path.isfile(self.log_output + ".img.ids"):
            with open(self.log_output + ".img.ids", 'r') as f:
                self.image_set: Dict[str, int] = {line.strip(): idx for idx, line in enumerate(f)}
        else:
            self.image_set: Dict[str, int] = {}

        self.image_count: Dict[str, int] = {}
        self.image_noid = 0

        self.entries_pattern = re.compile(r"(\n\n<\|[^>]+?\|>: )")
        self.header_pattern = re.compile(r"<\|([^@]+?)@(\d+)(?:\(([^)]+)\))?\|>:")
        self.cq_pattern = re.compile(r"\[CQ:(.*?)\]", re.DOTALL)
        self.cq_image_pattern = re.compile(r'\[CQ:image,id=(\d+)\]')

        self.escape_list = [
            ("&#91;", "["),
            ("&#93;", "]"),
            ("&#44;", ","),
            ("&amp;", "&"),
        ]

        self.user_blocklist = []
        self.cq_whitelist_1 = ["at", "face", "increase", "mface", "image"] # 会被基本原样保留的CQ码
        self.cq_whitelist_2 = self.cq_whitelist_1 + ["video", "file"] # 会被保留type的CQ码

        self.ctx = CTX
        self.pad_token = 65530
        self.message_separate = 24 # \x17

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
        for i in tqdm.trange(1, len(entries), 2, leave=False):
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
        print(f"parse_messages len(meaasges)", len(messages))
        return messages

    def load_obj(self, name):
        try:
            return joblib.load(self.obj_folder + name + ".gz")
        except:
            print(f"Load {name} from log")
            with open(self.log_folder + name, "r", encoding="utf-8") as f:
                input_text = f.read()
            obj = self.parse_messages(input_text)
            joblib.dump(obj, self.obj_folder + name + ".gz")
            return obj

    def wash_message(self, message: Message):
        if message.qq in self.user_blocklist:
            return None
        message.content = [
            cqobj
            for cqobj in message.content
            if (not isinstance(cqobj, CQObject)) or (cqobj.type in self.cq_whitelist_2)
        ]
        if len(message.content) == 0:
            return None
        return message

    def message_to_text(self, message: Message) -> str:
        """將Message對象轉換為處理後的文本"""
        parts = [f"<|{message.username}@{message.qq}{f'({message.group})' if message.group != "" else ""}|>: "]
        for part in message.content:
            if isinstance(part, str):
                parts.append(part.replace("\n", "\\n"))
            elif isinstance(part, CQObject):
                if part.type in self.cq_whitelist_1:
                    if part.type in ["image"]:
                        file = part.params.get("file", "")
                        
                        ## 统计+编号
                        if file in self.image_set:
                            image_id = self.image_set[file]
                        else:
                            image_id = -1
                            self.image_noid += 1

                        if file in self.image_count:
                            self.image_count[file]+=1
                        else:
                            self.image_count[file] = 1

                        parts.append(f"[CQ:image,id={image_id}]")
                        continue
                    cq_str = f"[CQ:{part.type}"
                    for k, v in part.params.items():
                        cq_str += f",{k}={self.cq_code_escape(v)}"
                    cq_str += "]"
                    parts.append(cq_str)
                else:
                    parts.append(f"[CQ:{part.type}]")        
        return "".join(parts)


    def process_group(self, messages: List[Message], n_tokens: int = 0) -> List[List[int]]:
        """处理消息并替换图片ID为分解后的token序列"""
        slices = []
        current_tokens = []
        global_blocks = []  # 保存所有未处理的块信息（起始索引，非padding长度）

        def decompose_image_id(image_id: int) -> List[int]:
            """分解image_id为32768进制（高位在前）"""
            ni = []
            while image_id > 0:
                image_id, rem = divmod(image_id, 32768)
                ni.insert(0, rem)
            return ni

        def process_text(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
            """返回处理后的token列表和块信息（相对起始位置，非padding长度）"""
            tokens = []
            blocks = []
            last_idx = 0

            prefix = ""
            for match in self.cq_image_pattern.finditer(text):
                # 处理匹配项前的文本
                pre_text = prefix + text[last_idx:match.start()] + "[CQ:image,"
                prefix = "]"
                tokens.extend(tokenizer.encode(pre_text))
                
                # 处理图片ID
                image_id = int(match.group(1))
                ni = decompose_image_id(image_id)
                non_pad_len = len(ni)
                ni_padded = ([self.pad_token] + ni + [self.pad_token] * 24)[:24]
                
                # 记录块信息
                block_start = len(tokens)
                blocks.append((block_start, non_pad_len))
                tokens.extend(ni_padded)
                last_idx = match.end()

            # 处理剩余文本
            post_text = text[last_idx:]
            tokens.extend(tokenizer.encode(post_text))
            return tokens, blocks

        for msg in tqdm.tqdm(messages, desc="dump&padding"):
            text = self.message_to_text(msg)
            #print(text)
            msg_tokens, msg_blocks = process_text(text)
            
            # 更新全局块索引
            prev_len = len(current_tokens)
            current_blocks = [
                (prev_len + start, non_pad_len)
                for (start, non_pad_len) in msg_blocks
            ]
            global_blocks.extend(current_blocks)
            
            current_tokens.extend(msg_tokens)

            current_tokens += [self.message_separate]

            # 处理分隔token
            if n_tokens > 0 and current_tokens:
                current_tokens += [self.pad_token] * n_tokens

            # 动态分片处理
            while len(current_tokens) >= self.ctx:
                # 检查需要替换的块
                replace_ranges = []
                for idx, (block_start, non_pad_len) in enumerate(global_blocks):
                    if block_start < self.ctx and (block_start + non_pad_len) > self.ctx:
                        replace_ranges.append((
                            block_start,
                            min(block_start + 24, len(current_tokens))  # 块总长24
                        ))

                # 替换被截断的块
                for start, end in replace_ranges:
                    for i in range(start, end):
                        if i < len(current_tokens):
                            current_tokens[i] = self.pad_token

                # 切片并保留剩余token
                slice_tokens = current_tokens[:self.ctx]
                slices.append(slice_tokens)
                current_tokens = current_tokens[self.ctx:]

                # 更新全局块索引
                new_global_blocks = []
                for block_start, non_pad_len in global_blocks:
                    new_start = block_start - self.ctx
                    if new_start >= 0:
                        new_global_blocks.append((new_start, non_pad_len))
                global_blocks = new_global_blocks

        # 处理最终剩余token
        if current_tokens:
            slices.append(current_tokens)
        print('Decoded text:', tokenizer.decodeBytes(slices[-1]).decode(errors="ignore"))
        return slices

    # def old_process_group(self, messages: List[Message], n_tokens: int = 0) -> List[List[int]]:
    #     """處理單個群組的消息並切片"""
    #     slices = []
    #     current_tokens = []
    #     pad_token = 65530
        
    #     for msg in messages:
    #         text = self.message_to_text(msg)
    #         msg_tokens = tokenizer.encode(text)
            
    #         if n_tokens > 0 and current_tokens:
    #             current_tokens += [pad_token] * n_tokens
            
    #         # 添加當前消息token
    #         current_tokens.extend(msg_tokens)
            
    #         # 切片處理
    #         while len(current_tokens) >= CTX:
    #             slice_tokens = current_tokens[:CTX]
    #             slices.append(slice_tokens)
    #             current_tokens = []
        
    #     return slices


    def athread(self, f):
        self.process_group(self.load_obj(f))

    def convert_all(self):
        os.makedirs(self.obj_folder, exist_ok=True)
        
        # 流式处理并立即写入
        builder = MMapIndexedDatasetBuilder(self.log_output+".bin", dtype=numpy.uint16)
        def process_pipeline():
            for f in os.listdir(self.log_folder):
                yield self.athread(f)
            # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            #     file_list = os.listdir(self.log_folder)
            #     for future in concurrent.futures.as_completed(
            #         [executor.submit(self.athread, f) for f in file_list]
            #     ):
            #         yield future.result()

            
        # 分批写入
        buffer = []
        buffer_size = 10000  # 根据内存调整批处理大小
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
        builder.finalize(self.log_output+".idx")
        print(f"{len(builder._sizes)}\n"*8) 
        print("image no id", self.image_noid)
        joblib.dump(self.image_count, self.log_output + ".img.count")
        

lp = LOG_Processer()
lp.convert_all()

# #预览数据集
# #预览数据集
# # 加载数据集
# dataset = MMapIndexedDataset('data/message_cache')
# print(f'Total documents: {len(dataset)}')

# # 解码前5条样本
# for i in range(5):
#     tokens = dataset[i].astype(int)
#     print(f'\nSample {i+1}:')
#     print('Token IDs:', tokens)
#     print('Decoded text:', tokenizer.decode(tokens.tolist()))
#     print('Token IDs:', tokens)