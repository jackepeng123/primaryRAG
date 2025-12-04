import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

KEYS = {
    "OpenAI": ["OPENAI_API_KEY"],
    "Qianfan (应用AK/SK)": ["QIANFAN_AK", "QIANFAN_SK"],
    "Qianfan (安全认证AK/SK)": ["QIANFAN_ACCESS_KEY", "QIANFAN_SECRET_KEY"],
    "IFLYTEK Spark": ["IFLYTEK_SPARK_APP_ID", "IFLYTEK_SPARK_API_KEY", "IFLYTEK_SPARK_API_SECRET"],
    "ZhipuAI": ["ZHIPUAI_API_KEY"],
}

def status(name, keys):
    vals = [bool(os.environ.get(k)) for k in keys]
    ok = all(vals)
    print(f"{name}: {'OK' if ok else 'MISSING'}")
    for k, v in zip(keys, vals):
        print(f"  - {k}: {'set' if v else 'unset'}")

if __name__ == "__main__":
    print("Environment keys status:\n")
    for name, keys in KEYS.items():
        status(name, keys)
    print("\nNote: 千帆免费模型包括 Yi-34B-Chat 等，付费/未开通模型将返回授权或额度错误。")

