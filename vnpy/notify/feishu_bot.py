import time
import hmac
import base64
import hashlib
import requests


class FeishuBot:
    def __init__(self, webhook_url: str, secret: str = ""):
        self.webhook_url = webhook_url
        self.secret = secret

    def _gen_sign(self, timestamp: int) -> str:
        """
        飞书自定义机器人签名
        """
        string_to_sign = f"{timestamp}\n{self.secret}"

        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()

        return base64.b64encode(hmac_code).decode("utf-8")

    def send_text(self, text: str):
        payload = {
            "msg_type": "text",
            "content": {
                "text": text
            }
        }

        if self.secret:
            timestamp = int(time.time())
            payload["timestamp"] = str(timestamp)
            payload["sign"] = self._gen_sign(timestamp)

        resp = requests.post(self.webhook_url,json=payload, timeout=5)

        if resp.status_code != 200:
            raise RuntimeError(f"飞书HTTP请求失败: status={resp.status_code}, body={resp.text}")
        result = resp.json()
        return result