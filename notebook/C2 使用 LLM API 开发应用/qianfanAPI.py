import os
import qianfan
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
def gen_wenxin_messages(prompt):
    messages = [{"role": "user", "content": prompt}]
    return messages

def get_completion(prompt, model="ERNIE-Bot", temperature=0.01):

    # text_1 = f"""
    # 泡一杯茶很容易。首先，需要把水烧开。\
    # 在等待期间，拿一个杯子并把茶包放进去。\
    # 一旦水足够热，就把它倒在茶包上。\
    # 等待一会儿，让茶叶浸泡。几分钟后，取出茶包。\
    # 如果您愿意，可以加一些糖或牛奶调味。\
    # 就这样，您可以享受一杯美味的茶了。
    # """

    # text_2 = f"""
    # 今天阳光明媚，鸟儿在歌唱。\
    # 这是一个去公园散步的美好日子。\
    # 鲜花盛开，树枝在微风中轻轻摇曳。\
    # 人们外出享受着这美好的天气，有些人在野餐，有些人在玩游戏或者在草地上放松。\
    # 这是一个完美的日子，可以在户外度过并欣赏大自然的美景。
    # """

    # prompt = f"""
    # 您将获得由三个引号括起来的文本。\
    # 如果它包含一系列的指令，则需要按照以下格式重新编写这些指令：
    # 第一步 - ...
    # 第二步 - …
    # …
    # 第N步 - …
    # 如果文本中不包含一系列的指令，则直接写“未提供步骤”。"
    # {text_2}
    # """

    text = f"""
    在一个迷人的村庄里，兄妹杰克和吉尔出发去一个山顶井里打水。\
    他们一边唱着欢乐的歌，一边往上爬，\
    然而不幸降临——杰克绊了一块石头，从山上滚了下来，吉尔紧随其后。\
    虽然略有些摔伤，但他们还是回到了温馨的家中。\
    尽管出了这样的意外，他们的冒险精神依然没有减弱，继续充满愉悦地探索。
    """

    prompt = f"""
    1-用一句话概括下面用<>括起来的文本。
    2-将摘要翻译成英语。
    3-在英语摘要中列出每个名称。
    4-输出一个 JSON 对象，其中包含以下键：English_summary，num_names。
    请使用以下格式（即冒号后的内容被<>括起来）：
    摘要：<摘要>
    翻译：<摘要的翻译>
    名称：<英语摘要中的名称列表>
    输出 JSON 格式：<带有 English_summary 和 num_names 的 JSON 格式>
    Text: <{text}>
    """

    chat_comp = qianfan.ChatCompletion()
    message = gen_wenxin_messages(prompt)

    resp = chat_comp.do(messages=message, 
                        model=model,
                        temperature = temperature)

    return resp["result"]

print(get_completion("你好，介绍一下你自己", model="ERNIE-Speed"))