from __future__ import annotations

from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import baostock as bs
import pandas as pd


@dataclass
class FinancialQueryResult:
    """
    单只股票某一年某一季度的财务因子结果。
    """
    code: str
    year: int
    quarter: int
    data: Dict[str, object]


class BaostockFinancialClient:
    """
    BaoStock 财务数据封装类

    主要能力：
    1. 登录 / 退出 BaoStock
    2. 查询盈利能力数据
    3. 查询成长能力数据
    4. 查询偿债能力数据
    5. 查询现金流数据
    6. 查询杜邦分析数据
    7. 合并成一行财务因子样本
    """

    def __init__(self, auto_login: bool = False) -> None:
        self._logged_in: bool = False

        if auto_login:
            self.login()

    def login(self) -> None:
        if self._logged_in:
            return

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {lg.error_code}, {lg.error_msg}")

        self._logged_in = True

    def logout(self) -> None:
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def __enter__(self) -> "BaostockFinancialClient":
        self.login()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.logout()

    # 转 DataFrame
    @staticmethod
    def _query_to_dataframe(rs) -> pd.DataFrame:
        """
        将 BaoStock 查询结果转换成 DataFrame
        """
        rows = []

        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock query failed: {rs.error_code}, {rs.error_msg}")

        return pd.DataFrame(rows, columns=rs.fields)

    # 转 float
    @staticmethod
    def _convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """
        尽量把数值列转成 float。
        code、pubDate、statDate 等字段保持字符串。
        """
        if df.empty:
            return df

        keep_string_cols = {
            "code",
            "pubDate",
            "statDate",
            "updateDate",
        }

        result = df.copy()

        for col in result.columns:
            if col in keep_string_cols:
                continue

            result[col] = pd.to_numeric(result[col], errors="ignore")

        return result

    # 查询盈利能力数据
    def query_profit_data(self, code: str, year: int, quarter: int) -> pd.DataFrame:
        """
        查询盈利能力数据。

        常见字段可能包括：
        - roeAvg: 净资产收益率
        - npMargin: 销售净利率
        - gpMargin: 销售毛利率
        - netProfit: 净利润
        - epsTTM: 每股收益
        """
        self.login()

        rs = bs.query_profit_data(code=code,year=year, quarter=quarter,)

        df = self._query_to_dataframe(rs)
        return self._convert_numeric(df)

    # 查询成长能力数据
    def query_growth_data(self, code: str, year: int, quarter: int) -> pd.DataFrame:
        """
        查询成长能力数据

        常见字段可能包括：
        - YOYEquity: 净资产同比增长率
        - YOYAsset: 总资产同比增长率
        - YOYNI: 净利润同比增长率
        - YOYEPSBasic: 基本每股收益同比增长率
        - YOYPNI: 归属母公司股东净利润同比增长率
        """
        self.login()

        rs = bs.query_growth_data(
            code=code,
            year=year,
            quarter=quarter,
        )

        df = self._query_to_dataframe(rs)
        return self._convert_numeric(df)

    # 查询偿债能力数据
    def query_balance_data(self, code: str, year: int, quarter: int) -> pd.DataFrame:
        """
        查询偿债能力数据

        常见字段可能包括：
        - currentRatio: 流动比率
        - quickRatio: 速动比率
        - cashRatio: 现金比率
        - YOYLiability: 总负债同比增长率
        - liabilityToAsset: 资产负债率
        """
        self.login()

        rs = bs.query_balance_data(
            code=code,
            year=year,
            quarter=quarter,
        )

        df = self._query_to_dataframe(rs)
        return self._convert_numeric(df)

    # 查询现金流数据
    def query_cash_flow_data(self, code: str, year: int, quarter: int) -> pd.DataFrame:
        """
        查询现金流数据。

        常见字段可能包括：
        - CAToAsset: 流动资产除以总资产
        - NCAToAsset: 非流动资产除以总资产
        - tangibleAssetToAsset: 有形资产除以总资产
        - ebitToInterest: 已获利息倍数
        - CFOToOR: 经营现金流除以营业收入
        - CFOToNP: 经营现金流除以净利润
        - CFOToGr: 经营现金流除以营业总收入
        """
        self.login()

        rs = bs.query_cash_flow_data(
            code=code,
            year=year,
            quarter=quarter,
        )

        df = self._query_to_dataframe(rs)
        return self._convert_numeric(df)

    # 查询杜邦分析数据
    def query_dupont_data(self, code: str, year: int, quarter: int) -> pd.DataFrame:
        """
        查询杜邦分析数据

        常见字段可能包括：
        - dupontROE: 杜邦净资产收益率
        - dupontAssetStoEquity: 权益乘数
        - dupontAssetTurn: 总资产周转率
        - dupontPnitoni: 归属母公司股东净利润 / 净利润
        - dupontNitogr: 净利润 / 营业总收入
        - dupontTaxBurden: 税负因子
        - dupontIntburden: 利息负担因子
        - dupontEbittogr: EBIT / 营业总收入
        """
        self.login()

        rs = bs.query_dupont_data(
            code=code,
            year=year,
            quarter=quarter,
        )

        df = self._query_to_dataframe(rs)
        return self._convert_numeric(df)

    # 合并数据
    @staticmethod
    def _dataframe_first_row_to_prefixed_dict(df: pd.DataFrame, prefix: str,) -> Dict[str, object]:
        """
        将 DataFrame 第一行转换成 dict 并给财务字段加前缀
        """
        if df.empty:
            return {}

        row = df.iloc[0].to_dict()

        common_keys = {
            "code",
            "pubDate",
            "statDate",
            "updateDate",
        }

        result: Dict[str, object] = {}

        for key, value in row.items():
            if key in common_keys:
                result[key] = value
            else:
                result[f"{prefix}_{key}"] = value

        return result

    # 查询并合并单只股票的财务因子
    def query_financial_factors(self, code: str, year: int, quarter: int,) -> FinancialQueryResult:
        """
        查询并合并单只股票的财务因子。

        返回结构：
        {
            "code": "sh.600000",
            "year": 2024,
            "quarter": 4,
            "profit_roeAvg": ...,
            "profit_npMargin": ...,
            "growth_YOYAsset": ...,
            "balance_liabilityToAsset": ...,
            "cash_CFOToNP": ...,
            "dupont_dupontROE": ...
        }
        """
        profit_df = self.query_profit_data(code, year, quarter)
        growth_df = self.query_growth_data(code, year, quarter)
        balance_df = self.query_balance_data(code, year, quarter)
        cash_df = self.query_cash_flow_data(code, year, quarter)
        dupont_df = self.query_dupont_data(code, year, quarter)

        data: Dict[str, object] = {
            "code": code,
            "year": year,
            "quarter": quarter,
        }

        data.update(self._dataframe_first_row_to_prefixed_dict(profit_df, "profit"))
        data.update(self._dataframe_first_row_to_prefixed_dict(growth_df, "growth"))
        data.update(self._dataframe_first_row_to_prefixed_dict(balance_df, "balance"))
        data.update(self._dataframe_first_row_to_prefixed_dict(cash_df, "cash"))
        data.update(self._dataframe_first_row_to_prefixed_dict(dupont_df, "dupont"))

        return FinancialQueryResult(
            code=code,
            year=year,
            quarter=quarter,
            data=data,
        )

    # 查询单只股票财务因子
    def query_financial_factors_df(self, code: str, year: int, quarter: int) -> pd.DataFrame:
        """
        查询单只股票财务因子，并返回 DataFrame。
        """
        result = self.query_financial_factors(code, year, quarter)
        return pd.DataFrame([result.data])

    #  批量查询多只股票的财务因子
    def query_financial_panel(self, codes: List[str], year: int, quarter: int, ignore_error: bool = True,) -> pd.DataFrame:
        """
        批量查询多只股票的财务因子。

        :param codes: 股票代码列表，例如 ["sh.600000", "sz.000001"]
        :param year: 年份
        :param quarter: 季度 1、2、3、4
        :param ignore_error: 单只股票失败时是否继续
        """
        rows: List[Dict[str, object]] = []

        for code in codes:
            try:
                result = self.query_financial_factors(code, year, quarter)
                rows.append(result.data)
            except Exception as e:
                if not ignore_error:
                    raise

                rows.append({
                    "code": code,
                    "year": year,
                    "quarter": quarter,
                    "error": str(e),
                })

        return pd.DataFrame(rows)
    
    # 获取指定日期的全部股票列表
    def query_all_stocks(self, query_date: str) -> pd.DataFrame:
        """
        获取 BaoStock 指定日期的全部证券列表。
        """
        self.login()

        rs = bs.query_all_stock(day=query_date)

        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock query_all_stock failed: {rs.error_code}, {rs.error_msg}")

        return self._query_to_dataframe(rs)

    # 获取指定日期的沪深 A 股股票列表
    def get_a_stocks(self, query_date: str, only_trading: bool = True) -> pd.DataFrame:
        """
        获取指定日期的沪深 A 股股票列表
        """
        import time

        start_time = time.perf_counter()

        df = self.query_all_stocks(query_date)

        if df.empty:
            print(
                f"[BaostockFinancialClient.get_a_stocks] return empty df, "
                f"query_date={query_date}",
                flush=True,
            )
            return df

        before_filter_count = len(df)

        a_stock_df = df[
            df["code"].str.match(
                r"^(sh\.(600|601|603|605|688)|sz\.(000|001|002|003|300|301))\d{3}$"
            )
        ].copy()


        if only_trading and "tradeStatus" in a_stock_df.columns:
            before_trading_filter_count = len(a_stock_df)

            a_stock_df = a_stock_df[a_stock_df["tradeStatus"] == "1"].copy()

        else:
            print(
                f"[BaostockFinancialClient.get_a_stocks] skip trading filter, "
                f"only_trading={only_trading}, "
                f"has_tradeStatus={'tradeStatus' in a_stock_df.columns}",
                flush=True,
            )

        result_df = a_stock_df.reset_index(drop=True)

        return result_df

    # 获取指定日期的上交所股票列表
    def get_sh_stocks(self, query_date: str) -> pd.DataFrame:
        """
        获取指定日期的上交所股票列表
        :param query_date: 查询日期，例如 '2026-05-13'
        """
        df = self.query_all_stocks(query_date)

        sh_df = df[df["code"].str.startswith("sh.")].copy()
        
        sh_a_df = df[
            df["code"].str.match(r"^sh\.(600|601|603|605|688)\d{3}$")
        ].copy()

        if "tradeStatus" in sh_a_df.columns:
            sh_a_df = sh_a_df[sh_a_df["tradeStatus"] == "1"].copy()

        return sh_df
    
    
class BaostockKlineClient:
    def __init__(self) -> None:
        pass

    def get_sh_a_stocks(self, query_date: str) -> pd.DataFrame:
        """
        获取指定日期的上交所A股股票列表
        """
        rs = bs.query_all_stock(day=query_date)

        df = self._query_to_dataframe(rs)

        if df.empty:
            return df

        # 只保留上交所A股
        sh_a_df = df[
            df["code"].str.match(r"^sh\.(600|601|603|605|688)\d{3}$")
        ].copy()

        # 只保留正常交易状态
        if "tradeStatus" in sh_a_df.columns:
            sh_a_df = sh_a_df[sh_a_df["tradeStatus"] == "1"].copy()

        return sh_a_df.reset_index(drop=True)
    
    # 将 BaoStock 查询结果转换成 DataFrame
    @staticmethod
    def _query_to_dataframe(rs) -> pd.DataFrame:
        """
        将 BaoStock 查询结果转换成 DataFrame。
        """
        rows = []

        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock query failed: {rs.error_code}, {rs.error_msg}")

        return pd.DataFrame(rows, columns=rs.fields)
    
    # 将K线里的数值字段转换成 float/int
    @staticmethod
    def _convert_kline_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """
        将K线里的数值字段转换成 float/int
        """
        if df.empty:
            return df

        result = df.copy()

        numeric_cols = ["open","high","low","close","preclose", "volume", "amount", "turn", "pctChg", "peTTM", "pbMRQ","psTTM", "pcfNcfTTM",]

        for col in numeric_cols:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        return result
    
    # 获取年的范围
    @staticmethod
    def get_recent_one_year_range(end_date: Optional[str] = None) -> tuple[str, str]:
        """
        获取最近一年日期范围。

        :param end_date: 结束日期，例如 '2026-05-13'。如果不传，默认今天。
        :return: (start_date, end_date)
        """
        if end_date is None:
            end_dt = datetime.today()
        else:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        start_dt = end_dt - timedelta(days=365)

        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    
    # 查询单只股票日K数据
    def query_daily_kline(self, code: str, start_date: str, end_date: str, adjustflag: str = "2",) -> pd.DataFrame:
        """
        查询单只股票日K数据

        :param code: 股票代码，例如 'sh.600000'
        :param start_date: 开始日期，例如 '2025-05-13'
        :param end_date: 结束日期，例如 '2026-05-13'
        :param adjustflag:
            1 后复权
            2 前复权
            3 不复权
        """

        fields = (
            "date,code,open,high,low,close,preclose,"
            "volume,amount,adjustflag,turn,tradestatus,"
            "pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
        )

        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, end_date=end_date, frequency="d", adjustflag=adjustflag)

        df = self._query_to_dataframe(rs)
        df = self._convert_kline_numeric(df)

        return df
    
    # 批量下载上交所A股日K数据
    def download_sh_a_daily_kline(
        self,
        query_date: str,
        start_date: str,
        end_date: str,
        output_dir: str = "data/sh_a_daily",
        adjustflag: str = "2",
        sleep_seconds: float = 0.2,
        overwrite: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        """
        批量下载上交所A股日K数据。
        """

        stocks = self.get_sh_a_stocks(query_date)

        if limit is not None:
            stocks = stocks.head(limit)

        total = len(stocks)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"准备下载上交所A股数量: {total}")
        print(f"日期范围: {start_date} ~ {end_date}")
        print(f"输出目录: {output_path}")

        success_count = 0
        skip_count = 0
        error_count = 0

        for idx, row in stocks.iterrows():
            code = row["code"]
            name = row.get("code_name", "")

            file_name = f"{code.replace('.', '_')}_{start_date}_{end_date}.csv"
            file_path = output_path / file_name

            print(f"\n[{idx + 1}/{total}] 处理 {code} {name}")

            if file_path.exists() and not overwrite:
                print(f"已存在，跳过: {file_path}")
                skip_count += 1
                continue

            try:
                df = self.query_daily_kline(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    adjustflag=adjustflag,
                )

                if df.empty:
                    print(f"无数据: {code}")
                    skip_count += 1
                else:
                    df.to_csv(file_path, index=False, encoding="utf-8-sig")
                    print(f"保存成功: {file_path}, 行数: {len(df)}")
                    success_count += 1

            except Exception as e:
                print(f"下载失败: {code}, error={e}")
                error_count += 1

            time.sleep(sleep_seconds)

        print("\n下载完成")
        print(f"成功: {success_count}")
        print(f"跳过: {skip_count}")
        print(f"失败: {error_count}")
    
    # 查询 BaoStock 行业分类数据
    def query_stock_industry(self) -> pd.DataFrame:
        """
        查询 BaoStock 行业分类数据

        返回字段通常包括：
            updateDate
            code
            code_name
            industry
            industryClassification
        """
        rs = bs.query_stock_industry()

        if rs.error_code != "0":
            raise RuntimeError(
                f"BaoStock query_stock_industry failed: "
                f"{rs.error_code}, {rs.error_msg}"
            )

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        df = pd.DataFrame(rows, columns=rs.fields)

        return df
