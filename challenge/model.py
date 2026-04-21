from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from xgboost import XGBClassifier


class DelayModel:
    FEATURES_COLS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    def __init__(self) -> None:
        self._model = None  # Model should be saved in this attribute.

    @staticmethod
    def _get_min_diff(data: pd.DataFrame) -> pd.Series:
        fecha_o = pd.to_datetime(data["Fecha-O"], errors="coerce")
        fecha_i = pd.to_datetime(data["Fecha-I"], errors="coerce")
        return ((fecha_o - fecha_i).dt.total_seconds() / 60).fillna(0)

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        opera_dummies = pd.get_dummies(data["OPERA"], prefix="OPERA")
        tipo_vuelo_dummies = pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO")
        mes_dummies = pd.get_dummies(data["MES"], prefix="MES")

        features = pd.concat(
            [opera_dummies, tipo_vuelo_dummies, mes_dummies],
            axis=1,
        )

        for column in self.FEATURES_COLS:
            if column not in features.columns:
                features[column] = 0

        return features[self.FEATURES_COLS].astype(int)

    @staticmethod
    def _default_training_data_path() -> Path:
        return Path(__file__).resolve().parents[1] / "data" / "data.csv"

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data = data.copy()
        features = self._build_features(data)

        if target_column is None:
            return features

        if target_column not in data.columns:
            if target_column != "delay":
                raise ValueError(f"Target column '{target_column}' not found in data.")
            data[target_column] = (self._get_min_diff(data) > 15).astype(int)

        target = data[[target_column]].copy().astype(int)
        return features, target

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        y = target.squeeze().astype(int)

        n_negative = int((y == 0).sum())
        n_positive = int((y == 1).sum())
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

        self._model = XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            n_estimators=300,
            max_depth=5,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        self._model.fit(features, y)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        training_data = pd.read_csv(self._default_training_data_path())
        features, target = self.preprocess(training_data, target_column="delay")
        self.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame,
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            List[int]: predicted targets.
        """
        self._ensure_model()
        predictions = self._model.predict(features)
        return [int(prediction) for prediction in predictions]