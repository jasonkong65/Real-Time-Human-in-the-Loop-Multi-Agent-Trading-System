import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class RiskAgent:
    """
    Risk Agent with a lightweight DQN-style advisory layer.

    It keeps the safety-first rule layer, but replaces the old Q-table decision
    logic with a small neural-network Q-value approximator. No extra deep
    learning package is required; sklearn's MLPRegressor is used as a compact
    DQN-style model for the assignment prototype.

    Important: the DQN layer is advisory only. Hard safety rules can block or
    soften a signal. The DQN cannot create a hard block by itself.
    """

    ACTIONS = ["KEEP_SIGNAL", "DOWNGRADE_TO_HOLD", "BLOCK_TRADE"]
    ACTION_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
    MODEL_SIGNALS = {"BUY_CANDIDATE": 1.0, "HOLD": 0.0, "SELL_RISK": -1.0, "BLOCKED": -0.5}

    def __init__(
        self,
        dqn_model_path: str = "models/risk_dqn_model.pkl",
        replay_path: str = "data/risk_dqn_replay.csv",
        q_table_path: str = "models/risk_q_table.pkl",
        epsilon: float = 0.08,
        gamma: float = 0.90,
        min_replay_samples: int = 8,
    ):
        self.dqn_model_path = Path(dqn_model_path)
        self.replay_path = Path(replay_path)
        self.q_table_path = Path(q_table_path)  # compatibility for older app/evaluator wording
        self.dqn_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.replay_path.parent.mkdir(parents=True, exist_ok=True)
        self.q_table_path.parent.mkdir(parents=True, exist_ok=True)
        self.epsilon = epsilon
        self.gamma = gamma
        self.min_replay_samples = min_replay_samples
        self.model = None
        self.scaler = None
        self.q_table = self._load_q_table_compat()
        self._load_dqn()

    # ------------------------------------------------------------------
    # Safe extraction helpers
    # ------------------------------------------------------------------
    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _clip(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _get_nested(self, data: Dict[str, Any], keys: List[str], default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    def _normalise_signal(self, value: Any, default: str = "HOLD") -> str:
        if value is None:
            return default
        value = str(value).strip().upper()
        return value if value else default

    def _get_symbol(self, signal_result: Dict[str, Any], analysis_result: Dict[str, Any], validation_result: Dict[str, Any]) -> str:
        for data in [signal_result, analysis_result, validation_result]:
            if isinstance(data, dict) and data.get("symbol"):
                return str(data.get("symbol")).upper().strip()
        nested = self._get_nested(signal_result, ["signal_for_next_agent", "symbol"])
        return str(nested or "UNKNOWN").upper().strip()

    def _validation_score(self, validation_result: Dict[str, Any]) -> float:
        score = self._safe_float(validation_result.get("confidence_score"))
        if score is not None:
            return self._clip(score)
        confidence = str(validation_result.get("confidence", "Medium")).lower()
        return {"high": 1.0, "medium": 0.75, "low": 0.45}.get(confidence, 0.6)

    def _validation_confidence(self, validation_result: Dict[str, Any]) -> str:
        return str(validation_result.get("confidence", "Medium")).title()

    def _validation_action(self, validation_result: Dict[str, Any]) -> str:
        return str(validation_result.get("next_action", "ALLOW_ANALYSIS")).upper()

    def _model_signal(self, signal_result: Dict[str, Any]) -> str:
        return self._normalise_signal(
            signal_result.get("model_signal") or self._get_nested(signal_result, ["signal_for_next_agent", "signal"]),
            "HOLD",
        )

    def _model_confidence(self, signal_result: Dict[str, Any]) -> float:
        value = self._safe_float(signal_result.get("prediction_confidence"))
        if value is None:
            value = self._safe_float(self._get_nested(signal_result, ["signal_for_next_agent", "prediction_confidence"]))
        return self._clip(value if value is not None else 0.5)

    def _model_confidence_level(self, signal_result: Dict[str, Any]) -> str:
        level = signal_result.get("confidence_level") or self._get_nested(signal_result, ["signal_for_next_agent", "confidence_level"])
        if level:
            return str(level).title()
        conf = self._model_confidence(signal_result)
        if conf >= 0.66:
            return "High"
        if conf >= 0.45:
            return "Medium"
        return "Low"

    def _analyst_signal(self, analysis_result: Dict[str, Any]) -> str:
        return self._normalise_signal(analysis_result.get("analyst_signal"), "NEUTRAL")

    def _analyst_score(self, analysis_result: Dict[str, Any]) -> float:
        return self._clip(self._safe_float(analysis_result.get("analyst_score"), 0.5) or 0.5)

    def _entry_risk(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        value = analysis_result.get("entry_risk_level") or stage2.get("entry_risk_level") or self._get_nested(signal_result, ["context_used", "entry_risk_level"])
        return str(value or "Medium").title()

    def _trend_direction(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        value = analysis_result.get("trend_direction") or stage2.get("trend_direction") or self._get_nested(signal_result, ["context_used", "trend_direction"])
        return str(value or "Neutral").title()

    def _volatility_level(self, analysis_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return str(analysis_result.get("volatility_level") or stage2.get("volatility_level") or "Unknown").title()

    def _feature_value(self, analysis_result: Dict[str, Any], key: str, default: float = 0.0) -> float:
        features = analysis_result.get("features_for_model", {}) if isinstance(analysis_result, dict) else {}
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return self._safe_float(features.get(key, stage2.get(key, default)), default) or default

    # ------------------------------------------------------------------
    # DQN state encoding
    # ------------------------------------------------------------------
    def _risk_numeric(self, label: str) -> float:
        label = str(label).lower()
        return {"low": 0.2, "medium": 0.55, "high": 0.85, "critical": 1.0, "unknown": 0.5}.get(label, 0.5)

    def _state_vector(self, validation_result: Dict[str, Any], analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> List[float]:
        model_signal = self._model_signal(signal_result)
        rsi = self._feature_value(analysis_result, "rsi_14", 50.0)
        return_20 = self._feature_value(analysis_result, "return_20", 0.0)
        ma_gap = self._feature_value(analysis_result, "ma_gap", 0.0)
        volatility_20 = self._feature_value(analysis_result, "volatility_20", 0.0)
        entry_risk = self._entry_risk(analysis_result, signal_result)
        trend = self._trend_direction(analysis_result, signal_result)
        volatility_level = self._volatility_level(analysis_result)
        analyst_signal = self._analyst_signal(analysis_result)
        return [
            self._validation_score(validation_result),
            self._model_confidence(signal_result),
            self._analyst_score(analysis_result),
            self.MODEL_SIGNALS.get(model_signal, 0.0),
            1.0 if "BULLISH" in analyst_signal or "POSITIVE" in analyst_signal else 0.0,
            1.0 if "BEARISH" in analyst_signal or "RISK" in analyst_signal else 0.0,
            self._risk_numeric(entry_risk),
            self._risk_numeric(volatility_level),
            1.0 if trend == "Positive" else (-1.0 if trend == "Negative" else 0.0),
            self._clip(rsi / 100.0),
            max(-1.0, min(1.0, return_20)),
            max(-1.0, min(1.0, ma_gap * 5.0)),
            self._clip(volatility_20 / 0.08),
        ]

    def _state_string(self, symbol: str, validation_result: Dict[str, Any], analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        parts = {
            "symbol": symbol,
            "validation": self._validation_confidence(validation_result),
            "model": self._model_signal(signal_result),
            "model_conf": self._model_confidence_level(signal_result),
            "analyst": self._analyst_signal(analysis_result),
            "trend": self._trend_direction(analysis_result, signal_result),
            "entry_risk": self._entry_risk(analysis_result, signal_result),
            "volatility": self._volatility_level(analysis_result),
        }
        return "|".join(f"{k}={v}" for k, v in parts.items())

    def _parse_state_string_to_vector(self, state: str) -> List[float]:
        # Used during delayed reward update when only the stored state string is available.
        data = {}
        for part in str(state or "").split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                data[k] = v
        validation = {"confidence": data.get("validation", "Medium")}
        signal = {"model_signal": data.get("model", "HOLD"), "confidence_level": data.get("model_conf", "Medium"), "prediction_confidence": {"High": 0.75, "Medium": 0.55, "Low": 0.35}.get(data.get("model_conf", "Medium"), 0.55)}
        analysis = {
            "analyst_signal": data.get("analyst", "NEUTRAL"),
            "analyst_score": 0.65 if "BULLISH" in data.get("analyst", "") or "POSITIVE" in data.get("analyst", "") else 0.45,
            "trend_direction": data.get("trend", "Neutral"),
            "entry_risk_level": data.get("entry_risk", "Medium"),
            "volatility_level": data.get("volatility", "Unknown"),
            "features_for_model": {"rsi_14": 50, "return_20": 0, "ma_gap": 0, "volatility_20": 0.02},
        }
        return self._state_vector(validation, analysis, signal)

    # ------------------------------------------------------------------
    # DQN storage and prediction
    # ------------------------------------------------------------------
    def _load_dqn(self) -> None:
        if not self.dqn_model_path.exists() or self.dqn_model_path.stat().st_size == 0:
            self.model = None
            self.scaler = None
            return
        try:
            bundle = joblib.load(self.dqn_model_path)
            if isinstance(bundle, dict):
                self.model = bundle.get("model")
                self.scaler = bundle.get("scaler")
            else:
                self.model = None
                self.scaler = None
        except Exception:
            self.model = None
            self.scaler = None

    def _save_dqn(self) -> None:
        try:
            joblib.dump({"model": self.model, "scaler": self.scaler, "actions": self.ACTIONS}, self.dqn_model_path)
        except Exception:
            pass

    def _load_replay(self) -> pd.DataFrame:
        if not self.replay_path.exists() or self.replay_path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(self.replay_path)
        except Exception:
            return pd.DataFrame()

    def _append_replay(self, state: str, vector: List[float], action: str, reward: float) -> None:
        row = {"state": state, "action": action, "reward": float(reward)}
        for i, value in enumerate(vector):
            row[f"x{i}"] = float(value)
        df = self._load_replay()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.replay_path, index=False)

    def _heuristic_q_values(self, vector: List[float]) -> Dict[str, float]:
        validation_score, model_conf, analyst_score, model_num = vector[0], vector[1], vector[2], vector[3]
        entry_risk, volatility_risk = vector[6], vector[7]
        trend = vector[8]
        keep = 0.15 + 0.25 * validation_score + 0.20 * model_conf + 0.15 * abs(model_num) + 0.10 * max(trend, 0)
        downgrade = 0.10 + 0.25 * entry_risk + 0.20 * volatility_risk + 0.15 * (1 - model_conf)
        block = 0.05 + 0.35 * (1 - validation_score) + 0.10 * max(0, model_num)
        if model_num < 0:
            keep += 0.15
        if analyst_score > 0.65 and trend > 0 and entry_risk > 0.65:
            downgrade += 0.20
        return {"KEEP_SIGNAL": keep, "DOWNGRADE_TO_HOLD": downgrade, "BLOCK_TRADE": block}

    def _predict_q_values(self, vector: List[float]) -> Dict[str, float]:
        if self.model is None or self.scaler is None:
            return self._heuristic_q_values(vector)
        try:
            X = self.scaler.transform(pd.DataFrame([vector]))
            values = self.model.predict(X)[0]
            return {action: float(values[i]) for i, action in enumerate(self.ACTIONS)}
        except Exception:
            return self._heuristic_q_values(vector)

    def _train_dqn_from_replay(self) -> Dict[str, Any]:
        df = self._load_replay()
        if df.empty:
            return {"trained": False, "reason": "No replay samples yet."}
        feature_cols = [c for c in df.columns if c.startswith("x")]
        if len(df) < self.min_replay_samples or not feature_cols:
            return {"trained": False, "reason": f"Need at least {self.min_replay_samples} replay samples."}

        X = df[feature_cols].astype(float)
        y_rows = []
        for _, row in df.iterrows():
            vec = [float(row[c]) for c in feature_cols]
            target = self._heuristic_q_values(vec)
            action = str(row.get("action", "KEEP_SIGNAL"))
            reward = self._safe_float(row.get("reward"), 0.0) or 0.0
            target[action if action in self.ACTIONS else "KEEP_SIGNAL"] = reward
            y_rows.append([target[a] for a in self.ACTIONS])
        y = pd.DataFrame(y_rows, columns=self.ACTIONS)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", max_iter=800, random_state=42)
        self.model.fit(X_scaled, y)
        self._save_dqn()
        return {"trained": True, "num_replay_samples": int(len(df)), "model_path": str(self.dqn_model_path)}

    # ------------------------------------------------------------------
    # Compatibility q-table summary
    # ------------------------------------------------------------------
    def _load_q_table_compat(self) -> Dict[str, Dict[str, float]]:
        if not self.q_table_path.exists() or self.q_table_path.stat().st_size == 0:
            return {}
        try:
            data = joblib.load(self.q_table_path)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_q_table_compat(self) -> None:
        try:
            joblib.dump(self.q_table, self.q_table_path)
        except Exception:
            pass

    def _record_state_q_values(self, state: str, q_values: Dict[str, float]) -> None:
        self.q_table[state] = {a: float(q_values.get(a, 0.0)) for a in self.ACTIONS}
        self._save_q_table_compat()

    # ------------------------------------------------------------------
    # Safety layer and decision combination
    # ------------------------------------------------------------------
    def _hard_safety_action(self, validation_result: Dict[str, Any], analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> Tuple[str, List[str]]:
        reasons = []
        validation_score = self._validation_score(validation_result)
        validation_action = self._validation_action(validation_result)
        model_signal = self._model_signal(signal_result)
        model_conf = self._model_confidence(signal_result)
        entry_risk = self._entry_risk(analysis_result, signal_result)
        volatility = self._volatility_level(analysis_result)

        if validation_action == "BLOCK_ANALYSIS" or validation_score <= 0.20:
            reasons.append("data quality is too weak")
            return "BLOCK_TRADE", reasons
        if model_signal == "BUY_CANDIDATE" and validation_score < 0.50:
            reasons.append("buy signal has weak data support")
            return "BLOCK_TRADE", reasons
        if model_signal == "BUY_CANDIDATE" and model_conf < 0.42:
            reasons.append("buy signal has low model confidence")
            return "DOWNGRADE_TO_HOLD", reasons
        if model_signal == "BUY_CANDIDATE" and entry_risk == "High":
            reasons.append("entry timing risk is high")
            return "DOWNGRADE_TO_HOLD", reasons
        if model_signal == "BUY_CANDIDATE" and volatility in ["High", "Critical"]:
            reasons.append("volatility is high")
            return "DOWNGRADE_TO_HOLD", reasons
        reasons.append("no hard safety block was triggered")
        return "KEEP_SIGNAL", reasons

    def _choose_dqn_action(self, q_values: Dict[str, float]) -> str:
        if random.random() < self.epsilon:
            return random.choice(["KEEP_SIGNAL", "DOWNGRADE_TO_HOLD"])
        return max(q_values, key=q_values.get)

    def _filter_dqn_action(self, dqn_action: str, hard_action: str, model_signal: str) -> str:
        if hard_action == "BLOCK_TRADE":
            return "BLOCK_TRADE"
        # The DQN cannot create a hard block alone in this educational system.
        if dqn_action == "BLOCK_TRADE":
            return "DOWNGRADE_TO_HOLD" if model_signal == "BUY_CANDIDATE" else "KEEP_SIGNAL"
        return dqn_action if dqn_action in self.ACTIONS else "KEEP_SIGNAL"

    def _combine_actions(self, hard_action: str, dqn_action: str) -> str:
        priority = {"KEEP_SIGNAL": 0, "DOWNGRADE_TO_HOLD": 1, "BLOCK_TRADE": 2}
        return hard_action if priority.get(hard_action, 0) >= priority.get(dqn_action, 0) else dqn_action

    def _apply_action(self, model_signal: str, action: str) -> str:
        if action == "BLOCK_TRADE":
            return "BLOCKED"
        if action == "DOWNGRADE_TO_HOLD" and model_signal == "BUY_CANDIDATE":
            return "HOLD"
        return model_signal if model_signal in ["BUY_CANDIDATE", "HOLD", "SELL_RISK"] else "HOLD"

    def _risk_level(self, final_signal: str, validation_result: Dict[str, Any], analysis_result: Dict[str, Any], signal_result: Dict[str, Any], action: str) -> Tuple[str, str]:
        points = 0
        notes = []
        if action == "BLOCK_TRADE":
            return "Critical", "Data quality or safety rules blocked the result."
        validation_score = self._validation_score(validation_result)
        model_conf = self._model_confidence(signal_result)
        entry_risk = self._entry_risk(analysis_result, signal_result)
        volatility = self._volatility_level(analysis_result)
        trend = self._trend_direction(analysis_result, signal_result)
        if validation_score < 0.55:
            points += 2
            notes.append("data confidence is limited")
        if model_conf < 0.45:
            points += 1
            notes.append("model confidence is not strong")
        if entry_risk == "High":
            points += 2
            notes.append("entry timing risk is high")
        elif entry_risk == "Medium":
            points += 1
            notes.append("entry timing risk is moderate")
        if volatility == "High":
            points += 2
            notes.append("volatility is high")
        if final_signal == "SELL_RISK":
            points += 2
            notes.append("the final signal points to downside risk")
        if trend == "Positive" and entry_risk in ["Medium", "High"] and final_signal == "HOLD":
            notes.append("main risk is chasing after a strong move")
        if points >= 5:
            level = "High"
        elif points >= 3:
            level = "Medium"
        else:
            level = "Low"
        return level, "; ".join(notes) if notes else "No major risk flag was detected."

    # ------------------------------------------------------------------
    # Main app-compatible method
    # ------------------------------------------------------------------
    def assess_risk(self, signal_result: Dict[str, Any], analysis_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        symbol = self._get_symbol(signal_result, analysis_result, validation_result)
        model_signal = self._model_signal(signal_result)
        vector = self._state_vector(validation_result, analysis_result, signal_result)
        state = self._state_string(symbol, validation_result, analysis_result, signal_result)
        hard_action, hard_reasons = self._hard_safety_action(validation_result, analysis_result, signal_result)
        q_values = self._predict_q_values(vector)
        raw_dqn_action = self._choose_dqn_action(q_values)
        filtered_dqn_action = self._filter_dqn_action(raw_dqn_action, hard_action, model_signal)
        final_action = self._combine_actions(hard_action, filtered_dqn_action)
        final_signal = self._apply_action(model_signal, final_action)
        risk_level, risk_interpretation = self._risk_level(final_signal, validation_result, analysis_result, signal_result, final_action)
        self._record_state_q_values(state, q_values)

        reasoning_steps = [
            f"Model signal: {model_signal}.",
            f"Validation confidence: {self._validation_confidence(validation_result)}.",
            f"Analyst signal: {self._analyst_signal(analysis_result)}.",
            f"Trend direction: {self._trend_direction(analysis_result, signal_result)}.",
            f"Entry timing risk: {self._entry_risk(analysis_result, signal_result)}.",
            f"Hard safety action: {hard_action} ({'; '.join(hard_reasons)}).",
            f"DQN advisory action: {raw_dqn_action}; after safety filter: {filtered_dqn_action}.",
            f"Final risk action: {final_action}; final signal: {final_signal}.",
        ]

        if final_action == "BLOCK_TRADE":
            decision = "The result was blocked because the safety layer found a serious data or risk issue."
        elif final_action == "DOWNGRADE_TO_HOLD":
            decision = "The signal was softened to HOLD because the setup needs more confirmation."
        else:
            decision = "The risk layer kept the model signal."

        return {
            "success": True,
            "agent": "Risk Agent",
            "agent_goal": "Apply hard safety checks and DQN-style risk control.",
            "symbol": symbol,
            "original_signal": model_signal,
            "risk_action": final_action,
            "hard_safety_action": hard_action,
            "dqn_action": raw_dqn_action,
            "filtered_dqn_action": filtered_dqn_action,
            "dqn_q_values": {k: round(v, 5) for k, v in q_values.items()},
            "dqn_model_path": str(self.dqn_model_path),
            "dqn_replay_path": str(self.replay_path),
            "q_state": state,
            "final_signal": final_signal,
            "risk_level": risk_level,
            "risk_interpretation": risk_interpretation,
            "volatility_level": self._volatility_level(analysis_result),
            "entry_risk_level": self._entry_risk(analysis_result, signal_result),
            "agent_decision": decision,
            "reasoning_steps": reasoning_steps,
            "risk_for_next_agent": {
                "symbol": symbol,
                "original_signal": model_signal,
                "final_signal": final_signal,
                "risk_level": risk_level,
                "risk_action": final_action,
                "dqn_action": filtered_dqn_action,
                "risk_interpretation": risk_interpretation,
                "explanation_for_llm": f"Risk level is {risk_level}. {risk_interpretation}",
            },
            "summary": f"Risk Agent set final signal to {final_signal} with {risk_level} risk.",
        }

    # Aliases expected by app.py
    def apply_risk_control(self, signal_result: Dict[str, Any], analysis_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    def adjust_risk(self, signal_result: Dict[str, Any], analysis_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    def evaluate_risk(self, signal_result: Dict[str, Any], analysis_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    def control_risk(self, signal_result: Dict[str, Any], analysis_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    def run(self, signal_result: Dict[str, Any], analysis_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    # ------------------------------------------------------------------
    # Delayed reward learning
    # ------------------------------------------------------------------
    def calculate_reward(self, final_signal: str, future_return: float, volatility_level: str = "Unknown") -> float:
        future_return = self._safe_float(future_return, 0.0) or 0.0
        final_signal = self._normalise_signal(final_signal, "HOLD")
        if final_signal == "BUY_CANDIDATE":
            reward = future_return
        elif final_signal == "SELL_RISK":
            reward = -future_return
        elif final_signal == "HOLD":
            # HOLD is good if it avoids a large move against the signal, but it
            # should not dominate learning.
            reward = -abs(future_return) * 0.15
        elif final_signal == "BLOCKED":
            reward = abs(future_return) * 0.3 if future_return < 0 else -future_return * 0.2
        else:
            reward = 0.0
        if str(volatility_level).title() in ["High", "Critical"]:
            reward -= 0.003
        return float(reward)

    def update_q_value(self, state: str, action: str, reward: float, next_state: Optional[str] = None) -> Dict[str, Any]:
        # The name is kept for app.py/reward_agent compatibility, but the update
        # now writes DQN replay memory and retrains the small neural Q model.
        if not state:
            return {"success": False, "summary": "Cannot update DQN because state is missing."}
        action = action if action in self.ACTIONS else "KEEP_SIGNAL"
        reward = self._safe_float(reward, 0.0) or 0.0
        vector = self._parse_state_string_to_vector(state)
        old_q = self._predict_q_values(vector).get(action, 0.0)
        self._append_replay(state, vector, action, reward)
        train_result = self._train_dqn_from_replay()
        new_q = self._predict_q_values(vector).get(action, old_q)
        q_values = self._predict_q_values(vector)
        self._record_state_q_values(state, q_values)
        return {
            "success": True,
            "learning_type": "DQN-style replay update",
            "state": state,
            "action": action,
            "reward": round(reward, 6),
            "old_q": round(old_q, 6),
            "new_q": round(new_q, 6),
            "dqn_model_path": str(self.dqn_model_path),
            "dqn_replay_path": str(self.replay_path),
            "train_result": train_result,
            "summary": "Updated the DQN replay memory and refreshed the risk advisory model when enough samples were available.",
        }

    def update_from_feedback(self, risk_result: Dict[str, Any], future_return: float) -> Dict[str, Any]:
        if not isinstance(risk_result, dict):
            return {"success": False, "summary": "Cannot update DQN because risk_result is invalid."}
        final_signal = risk_result.get("final_signal")
        risk_level = risk_result.get("risk_level")
        volatility = risk_result.get("volatility_level") or ("High" if risk_level in ["High", "Critical"] else "Low")
        reward = self.calculate_reward(final_signal, future_return, volatility)
        result = self.update_q_value(
            state=risk_result.get("q_state"),
            action=risk_result.get("risk_action"),
            reward=reward,
        )
        result.update({
            "final_signal": final_signal,
            "risk_level": risk_level,
            "future_return": self._safe_float(future_return, 0.0),
            "calculated_reward": round(reward, 6),
        })
        return result
