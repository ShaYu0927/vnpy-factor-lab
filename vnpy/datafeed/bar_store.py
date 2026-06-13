

from pathlib import Path
import pandas as pd
from typing import List
from pyparsing import Optional



class FactorStore:
    """
    因子数据存储器。
    """

    def __init__(self, root_dir: str = "data/factors"):
        """
        初始化因子存储路径。

        Parameters
        ----------
        root_dir : str
            因子数据根目录。
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_factor(self, symbol: str, factor_name: str, factor_value: float):
        """
        保存某只股票的某个因子值
        """
        factor_file = self.root_dir / f"{symbol}_{factor_name}.txt"
        factor_file.write_text(str(factor_value))

    def get_factor(self, symbol: str, factor_name: str) -> float:
        """
        获取某只股票的某个因子值
        """
        return self._store.get(symbol, {}).get(factor_name, None)
    
    def _get_factor_path(self, factor_name: str) -> Path:
        """
        获取某个因子的文件路径
        """
        return self.root_dir / f"{factor_name}.parquet"
    
    def load_factor(self, factor_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        读取因子数据。

        Parameters
        ----------
        factor_name : str
            因子名称。
        start_date : Optional[str]
            开始日期，例如 2026-04-01。
        end_date : Optional[str]
            结束日期，例如 2026-04-29。
        symbols : Optional[List[str]]
            股票代码列表，例如 ["SHSE.600519", "SZSE.000001"]。
        Returns
        -------
        pd.DataFrame
            因子数据。
        """
        file_path = self._get_factor_path(factor_name)

        if not file_path.exists():
            raise FileNotFoundError(f"因子文件不存在: {file_path}")

        df = pd.read_parquet(file_path)
        if start_date is not None:
            df = df[df["trade_date"] >= start_date]
        if end_date is not None:
            df = df[df["trade_date"] <= end_date]
        if symbols is not None:
            df = df[df["symbol"].isin(symbols)]

        return df.reset_index(drop=True)
    
    def exists(self, factor_name: str) -> bool:
        """
        判断某个因子文件是否存在。
        """
        return self._get_factor_path(factor_name).exists()
    
    def delete_factor(self, factor_name: str) -> None:
        """
        删除某个因子文件。
        """
        file_path = self._get_factor_path(factor_name)

        if file_path.exists():
            file_path.unlink()
            
    def list_factors(self) -> List[str]:
        """
        查看当前已经保存了哪些因子。
        """
        files = self.root_dir.glob("*.parquet")
        return [file.stem for file in files]
    
    def merge_factors(
        self,
        factor_names: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        读取并合并多个因子。

        例如：
        momentum_20 + volatility_20 + turnover_20

        合并后结果：

        trade_date      symbol        momentum_20     volatility_20
        2026-04-29      SHSE.600519   0.052           0.018
        """
        if not factor_names:
            raise ValueError("factor_names 不能为空")

        result_df = None

        for factor_name in factor_names:
            df = self.load_factor(
                factor_name=factor_name,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
            )

            if result_df is None:
                result_df = df
            else:
                result_df = result_df.merge(
                    df,
                    on=["trade_date", "symbol"],
                    how="inner",
                )

        return result_df.reset_index(drop=True)
    
    def append_factor(
        self,
        factor_name: str,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
    ) -> Path:
        """
        追加保存因子数据
        """
        self._check_factor_df(df)

        file_path = self._get_factor_path(factor_name)

        if file_path.exists():
            old_df = pd.read_parquet(file_path)
            new_df = pd.concat([old_df, df], ignore_index=True)

            if drop_duplicates:
                new_df = new_df.drop_duplicates(
                    subset=["trade_date", "symbol"],
                    keep="last",
                )
        else:
            new_df = df

        new_df.to_parquet(file_path, index=False)

        return file_path
    
    def _check_factor_df(self, df: pd.DataFrame) -> None:
        """
        检查因子 DataFrame 是否符合基本要求。
        """
        required_columns = {"trade_date", "symbol"}

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(f"因子数据缺少必要字段: {missing_columns}")

        if df.empty:
            raise ValueError("因子数据不能为空")

        if df["trade_date"].isnull().any():
            raise ValueError("trade_date 存在空值")

        if df["symbol"].isnull().any():
            raise ValueError("symbol 存在空值")

