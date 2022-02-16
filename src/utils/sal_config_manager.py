"""ESN全体の学習を行う際の SAL に関する学習設定を管理するコードを実装する場所"""
import logging
import os
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_LIST = [
    "dims", "beta", "leaky_rate", "sample_rate", "n_mels", "chunk", "n_fft",
    "input_scale", "input_offset", "num_concat"
]
PATH_NAME = "save_path"


class SALConfigManager(object):
    """SAL で学習させたネットワーク情報を管理する class"""
    def __init__(self, config_master_file: str):
        self.config_master_file = config_master_file
        if os.path.exists(self.config_master_file):
            self.config = pd.read_csv(config_master_file)
        else:
            self.config = pd.DataFrame(columns=CONFIG_LIST)

    def search_config(self, configs: Dict):
        """SAL の学習設定で同様のものがあるかどうかを探す

        Parameters
        ----------
        configs : Dict
            CONFIG_LIST の中身が入っていることが期待される
        """
        query_parts = []
        for key in CONFIG_LIST:
            if key == PATH_NAME:
                continue
            elif key == "dims":
                query_parts.append(f'{key}=="{configs[key]}"')
            else:
                query_parts.append(f'{key}=={configs[key]}')
        query_string = " & ".join(query_parts)
        queried_items = self.config.query(query_string)
        if queried_items.shape[0] > 0:
            return queried_items[PATH_NAME][0]
        else:
            return None

    def add_config(self, config: Dict):
        """学習させたネットワークの config を追加して csv に出力する

        Parameters
        ----------
        config : Dict
            [description]
        """
        new_item = {}
        for key in CONFIG_LIST:
            new_item[key] = config[key]
        new_item[PATH_NAME] = config[PATH_NAME]
        self.config = self.config.append(new_item, ignore_index=True)
        self.config.to_csv(self.config_master_file, index=None)
