

import base64
from dataclasses import dataclass
import hashlib
import hmac
import time
from typing import Optional

import requests

@dataclass
class KlineSnapshot:
    """
    A single candlestick bar or one K-line analysis result.
    """
    symbol: str
    period: str
    close: float
    prev_close: float
    volume: float
    volume_ma5: float
    ma5: float
    ma10: float
    
class FeishuWebhookBot:
    """
    飞书群自定义机器人 Webhook 推送
    """

    def __init__(self, webhook_url: str, secret: str = ""):
        self.webhook_url = webhook_url
        self.secret = secret

    def _gen_sign(self, timestamp: int) -> str:
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(string_to_sign.encode("utf-8"),b"",digestmod=hashlib.sha256,).digest()
        return base64.b64encode(hmac_code).decode("utf-8")

    def send_text(self, text: str) -> dict:
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

        resp = requests.post(
            self.webhook_url,
            json=payload,
            timeout=10,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"飞书推送HTTP失败: status={resp.status_code}, body={resp.text}"
            )

        result = resp.json()

        if result.get("code", 0) != 0:
            raise RuntimeError(f"飞书推送业务失败: {result}")

        return result
    
class KlineMessageBuilder:
    """
    K线消息构造器
    """
    @staticmethod
    def calc_ret(close: float, prev_close: float) -> float:
        if prev_close <= 0:
            return 0.0
        return close / prev_close - 1

    @staticmethod
    def calc_volume_ratio(volume: float, volume_ma5: float) -> float:
        if volume_ma5 <= 0:
            return 0.0
        return volume / volume_ma5

    @classmethod
    def judge_signal(cls, kline: KlineSnapshot) -> tuple[str, str]:
        volume_ratio = cls.calc_volume_ratio(kline.volume,kline.volume_ma5)

        if kline.close > kline.ma5 > kline.ma10 and volume_ratio >= 2:
            return ("放量上涨","价格站上MA5、MA10,且成交量超过5周期均量2倍")

        if kline.close < kline.ma5 < kline.ma10 and volume_ratio >= 2:
            return ("放量下跌","价格跌破MA5、MA10,且成交量超过5周期均量2倍")

        if kline.close > kline.ma5 > kline.ma10:
            return ("均线多头","价格位于MA5上方,且MA5高于MA10")

        if kline.close < kline.ma5 < kline.ma10:
            return ("均线空头", "价格位于MA5下方,且MA5低于MA10")

        return ("普通波动", "暂无明显放量或均线信号")

    @classmethod
    def build(cls, kline: KlineSnapshot) -> str:
        ret = cls.calc_ret(kline.close, kline.prev_close)
        volume_ratio = cls.calc_volume_ratio(kline.volume, kline.volume_ma5)
        signal, reason = cls.judge_signal(kline)

        return f"""
            【股票K线提醒】

            股票：{kline.symbol}
            周期：{kline.period}
            最新价：{kline.close:.2f}
            涨跌幅：{ret * 100:.2f}%

            成交量：{kline.volume:.0f}
            5周期均量: {kline.volume_ma5:.0f}
            量比：{volume_ratio:.2f}

            MA5:{kline.ma5:.2f}
            MA10:{kline.ma10:.2f}

            信号:{signal}
            原因:{reason}
            """.strip()

    
class KlineFeishuPusher:
    """
    股票K线飞书推送器
    """

    def __init__(
        self,
        webhook_url: str,
        secret: str = "",
        push_all: bool = False,
        min_ret: float = 0.01,
        min_volume_ratio: float = 2.0,
        cooldown_seconds: int = 60,
    ):
        self.bot = FeishuWebhookBot(webhook_url, secret)
        self.push_all = push_all
        self.min_ret = min_ret
        self.min_volume_ratio = min_volume_ratio
        self.cooldown_seconds = cooldown_seconds
        self._last_push_time = {}

        
        
    def should_push(self, kline: KlineSnapshot) -> bool:
        if self.push_all:
            return True

        ret = KlineMessageBuilder.calc_ret(kline.close, kline.prev_close)
        volume_ratio = KlineMessageBuilder.calc_volume_ratio(kline.volume, kline.volume_ma5)

        signal, _ = KlineMessageBuilder.judge_signal(kline)

        hit_signal = signal != "普通波动"
        hit_ret = abs(ret) >= self.min_ret
        hit_volume = volume_ratio >= self.min_volume_ratio

        if not (hit_signal or hit_ret or hit_volume):
            return False

        now = time.time()
        last_time = self._last_push_time.get(kline.symbol, 0)

        if now - last_time < self.cooldown_seconds:
            return False

        self._last_push_time[kline.symbol] = now
        return True

    def push_force(self, kline: KlineSnapshot) -> dict:
        """
        forch push
        """
        msg = KlineMessageBuilder.build(kline)
        return self.bot.send_text(msg)
    
    def push(self, kline: KlineSnapshot) -> Optional[dict]:
        """
        判断是否需要推送，需要则发送到飞书
        """
        if not self.should_push(kline):
            return None

        msg = KlineMessageBuilder.build(kline)
        return self.bot.send_text(msg)
