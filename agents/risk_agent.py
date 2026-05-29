import json
import math
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """
    Small feed-forward Q-network used by RiskAgent.

    Input: numeric risk state vector.
    Output: Q-values for [KEEP_SIGNAL, DOWNGRADE_TO_HOLD, BLOCK_TRADE].
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RiskAgent:
    """
    Strict DQN Risk Agent.

    Role:
    - Hard rule-based safety layer decides non-negotiable safety constraints.
    - PyTorch DQN advisory layer learns risk actions from delayed paper rewards.
    - Target network + replay memory make it a real DQN implementation, not only
      a DQN-style sklearn approximator.

    Safety design:
    - DQN is advisory. It can increase caution for risky buy setups, but it cannot
      create BLOCK_TRADE by itself unless hard safety already triggered a block.
    - This keeps the project suitable for paper decision support, not real trading.
    """

    ACTIONS = ["KEEP_SIGNAL", "DOWNGRADE_TO_HOLD", "BLOCK_TRADE"]
    ACTION_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
    INDEX_ACTION = {idx: name for name, idx in ACTION_INDEX.items()}

    MODEL_SIGNALS = {
        "BUY_CANDIDATE": 1.0,
        "HOLD": 0.0,
        "SELL_RISK": -1.0,
        "BLOCKED": -0.5,
        "BUY_WATCHLIST_OVERBOUGHT": 0.6,
        "BUY_WATCHLIST_ENTRY_RISK": 0.6,
        "WATCHLIST_BULLISH_ENTRY_RISK": 0.6,
    }

    STATE_DIM = 13


    def _load_runtime_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load optional DQN/risk settings from config/risk_config.json.
        This keeps the agent usable without manual code edits while preserving
        backward-compatible constructor arguments.
        """
        candidate_paths = []
        if config_path:
            candidate_paths.append(Path(config_path))
        candidate_paths.append(Path("config/risk_config.json"))

        for path in candidate_paths:
            try:
                if path.exists() and path.stat().st_size > 0:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                continue
        return {}

    def _cfg(self, config: Dict[str, Any], key: str, default: Any) -> Any:
        """Read one config value from either the top level or the dqn section."""
        try:
            if key in config:
                return config[key]
            dqn = config.get("dqn", {}) if isinstance(config, dict) else {}
            return dqn.get(key, default) if isinstance(dqn, dict) else default
        except Exception:
            return default

    def __init__(
        self,
        dqn_model_path: str = "models/risk_dqn_model.pt",
        target_model_path: str = "models/risk_dqn_target_model.pt",
        replay_path: str = "data/risk_dqn_replay.csv",
        q_table_path: str = "models/risk_q_table.pkl",
        epsilon: float = 0.08,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        gamma: float = 0.90,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        min_replay_samples: int = 100,
        target_update_steps: int = 25,
        train_epochs_per_update: int = 4,
        hidden_dim: int = 64,
        seed: int = 42,
        config_path: Optional[str] = "config/risk_config.json",
        replay_db_path: str = "data/trading_system.db",
    ):
        config = self._load_runtime_config(config_path)
        epsilon = float(self._cfg(config, "epsilon", epsilon))
        epsilon_min = float(self._cfg(config, "epsilon_min", epsilon_min))
        epsilon_decay = float(self._cfg(config, "epsilon_decay", epsilon_decay))
        gamma = float(self._cfg(config, "gamma", gamma))
        learning_rate = float(self._cfg(config, "learning_rate", learning_rate))
        batch_size = int(self._cfg(config, "batch_size", batch_size))
        min_replay_samples = int(self._cfg(config, "min_replay_samples", min_replay_samples))
        target_update_steps = int(self._cfg(config, "target_update_steps", target_update_steps))
        train_epochs_per_update = int(self._cfg(config, "train_epochs_per_update", train_epochs_per_update))
        hidden_dim = int(self._cfg(config, "hidden_dim", hidden_dim))
        replay_db_path = str(self._cfg(config, "replay_db_path", replay_db_path))

        self.dqn_model_path = Path(dqn_model_path)
        self.target_model_path = Path(target_model_path)
        self.replay_path = Path(replay_path)
        self.q_table_path = Path(q_table_path)  # compatibility for older app/evaluator wording
        self.replay_db_path = Path(replay_db_path)

        self.dqn_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.target_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.replay_path.parent.mkdir(parents=True, exist_ok=True)
        self.q_table_path.parent.mkdir(parents=True, exist_ok=True)
        self.replay_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.loss_history_path = Path("data/risk_dqn_loss_history.csv")
        self.loss_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_training_loss: Optional[float] = None

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_replay_samples = min_replay_samples
        self.target_update_steps = target_update_steps
        self.train_epochs_per_update = train_epochs_per_update
        self.hidden_dim = hidden_dim
        self.seed = seed

        random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(self.STATE_DIM, len(self.ACTIONS), hidden_dim).to(self.device)
        self.target_net = DQNNetwork(self.STATE_DIM, len(self.ACTIONS), hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.training_steps = 0

        self.q_table = self._load_q_table_compat()
        self._init_replay_db()
        self._load_or_init_dqn()

    # ------------------------------------------------------------------
    # Safe extraction helpers
    # ------------------------------------------------------------------
    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            if isinstance(value, str) and value.lower() in ["none", "nan", "null"]:
                return default
            output = float(value)
            if math.isnan(output) or math.isinf(output):
                return default
            return output
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

    def _normalise_label(self, value: Any, default: str = "UNKNOWN") -> str:
        if value is None:
            return default
        value = str(value).strip().upper()
        return value if value else default

    def _get_symbol(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        validation_result: Dict[str, Any],
    ) -> str:
        candidates = [
            signal_result.get("symbol") if isinstance(signal_result, dict) else None,
            self._get_nested(signal_result, ["signal_for_next_agent", "symbol"]),
            analysis_result.get("symbol") if isinstance(analysis_result, dict) else None,
            validation_result.get("symbol") if isinstance(validation_result, dict) else None,
            self._get_nested(validation_result, ["validation_for_next_agent", "symbol"]),
        ]
        for item in candidates:
            if item:
                return str(item).upper().strip()
        return "UNKNOWN"

    def _validation_score(self, validation_result: Dict[str, Any]) -> float:
        score = self._safe_float(validation_result.get("confidence_score"))
        if score is not None:
            return self._clip(score)
        confidence = str(validation_result.get("confidence", "Medium")).lower()
        return {"high": 1.0, "medium": 0.72, "low": 0.40}.get(confidence, 0.60)

    def _validation_confidence(self, validation_result: Dict[str, Any]) -> str:
        return str(validation_result.get("confidence", "Medium")).title()

    def _validation_action(self, validation_result: Dict[str, Any]) -> str:
        return str(validation_result.get("next_action", "ALLOW_ANALYSIS")).upper()

    def _model_signal(self, signal_result: Dict[str, Any]) -> str:
        value = (
            signal_result.get("model_signal")
            or signal_result.get("final_signal")
            or signal_result.get("display_signal")
            or self._get_nested(signal_result, ["signal_for_next_agent", "signal"])
            or "HOLD"
        )
        return self._normalise_label(value, "HOLD")

    def _model_confidence(self, signal_result: Dict[str, Any]) -> float:
        value = self._safe_float(signal_result.get("prediction_confidence"))
        if value is None:
            value = self._safe_float(self._get_nested(signal_result, ["signal_for_next_agent", "prediction_confidence"]))
        return self._clip(value if value is not None else 0.50)

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
        return self._normalise_label(analysis_result.get("analyst_signal"), "NEUTRAL")

    def _analyst_score(self, analysis_result: Dict[str, Any]) -> float:
        return self._clip(self._safe_float(analysis_result.get("analyst_score"), 0.50) or 0.50)

    def _entry_risk(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        value = (
            analysis_result.get("entry_risk_level")
            or stage2.get("entry_risk_level")
            or self._get_nested(signal_result, ["context_used", "entry_risk_level"])
            or "Medium"
        )
        return str(value).title()

    def _trend_direction(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        value = (
            analysis_result.get("trend_direction")
            or stage2.get("trend_direction")
            or self._get_nested(signal_result, ["context_used", "trend_direction"])
            or "Neutral"
        )
        return str(value).title()

    def _volatility_level(self, analysis_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        value = analysis_result.get("volatility_level") or stage2.get("volatility_level") or "Unknown"
        return str(value).title()

    def _feature_value(self, analysis_result: Dict[str, Any], key: str, default: float = 0.0) -> float:
        features = analysis_result.get("features_for_model", {}) if isinstance(analysis_result, dict) else {}
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return self._safe_float(features.get(key, stage2.get(key, default)), default) or default

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------
    def _risk_numeric(self, label: str) -> float:
        label = str(label).lower()
        return {
            "low": 0.20,
            "medium": 0.55,
            "moderate": 0.55,
            "high": 0.85,
            "critical": 1.0,
            "unknown": 0.50,
        }.get(label, 0.50)

    def _trend_numeric(self, trend: str) -> float:
        trend = str(trend).lower()
        if "strong positive" in trend:
            return 1.0
        if "positive" in trend:
            return 0.65
        if "strong negative" in trend:
            return -1.0
        if "negative" in trend:
            return -0.65
        return 0.0

    def _state_vector(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> List[float]:
        model_signal = self._model_signal(signal_result)
        analyst_signal = self._analyst_signal(analysis_result)
        entry_risk = self._entry_risk(analysis_result, signal_result)
        trend = self._trend_direction(analysis_result, signal_result)
        volatility_level = self._volatility_level(analysis_result)

        rsi = self._feature_value(analysis_result, "rsi_14", 50.0)
        return_5 = self._feature_value(analysis_result, "return_5", 0.0)
        return_20 = self._feature_value(analysis_result, "return_20", 0.0)
        ma_gap = self._feature_value(analysis_result, "ma_gap", 0.0)
        volatility_20 = self._feature_value(analysis_result, "volatility_20", 0.0)

        return [
            self._validation_score(validation_result),
            self._model_confidence(signal_result),
            self._analyst_score(analysis_result),
            self.MODEL_SIGNALS.get(model_signal, 0.0),
            1.0 if ("BULLISH" in analyst_signal or "POSITIVE" in analyst_signal) else 0.0,
            1.0 if ("BEARISH" in analyst_signal or analyst_signal == "SELL_RISK") else 0.0,
            self._risk_numeric(entry_risk),
            self._risk_numeric(volatility_level),
            self._trend_numeric(trend),
            self._clip(rsi / 100.0),
            max(-1.0, min(1.0, return_5)),
            max(-1.0, min(1.0, return_20)),
            max(-1.0, min(1.0, ma_gap * 5.0 + volatility_20)),
        ]

    def _state_string(
        self,
        symbol: str,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> str:
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
        """
        Compatibility path for delayed RewardAgent updates.
        It reconstructs a reasonable vector from the stored q_state string.
        """
        parsed = {}
        for part in str(state or "").split("|"):
            if "=" in part:
                key, value = part.split("=", 1)
                parsed[key.strip()] = value.strip()

        validation = parsed.get("validation", "Medium").title()
        model = parsed.get("model", "HOLD").upper()
        model_conf = parsed.get("model_conf", "Medium").title()
        analyst = parsed.get("analyst", "NEUTRAL").upper()
        trend = parsed.get("trend", "Neutral").title()
        entry_risk = parsed.get("entry_risk", "Medium").title()
        volatility = parsed.get("volatility", "Unknown").title()

        validation_score = {"High": 1.0, "Medium": 0.72, "Low": 0.40}.get(validation, 0.60)
        model_conf_score = {"High": 0.80, "Medium": 0.55, "Low": 0.32}.get(model_conf, 0.50)
        analyst_score = 0.70 if ("BULLISH" in analyst or "POSITIVE" in analyst) else 0.35 if "BEARISH" in analyst else 0.50

        return [
            validation_score,
            model_conf_score,
            analyst_score,
            self.MODEL_SIGNALS.get(model, 0.0),
            1.0 if ("BULLISH" in analyst or "POSITIVE" in analyst) else 0.0,
            1.0 if ("BEARISH" in analyst or analyst == "SELL_RISK") else 0.0,
            self._risk_numeric(entry_risk),
            self._risk_numeric(volatility),
            self._trend_numeric(trend),
            0.50,
            0.0,
            0.0,
            0.0,
        ]

    # ------------------------------------------------------------------
    # DQN persistence
    # ------------------------------------------------------------------
    def _load_q_table_compat(self) -> Dict[str, Dict[str, float]]:
        if self.q_table_path.exists() and self.q_table_path.stat().st_size > 0:
            try:
                data = joblib.load(self.q_table_path)
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        return {}

    def _save_q_table_compat(self) -> None:
        try:
            joblib.dump(self.q_table, self.q_table_path)
        except Exception:
            pass

    def _load_or_init_dqn(self) -> None:
        checkpoint_path = self.dqn_model_path
        if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint.get("policy_state_dict", checkpoint))
                if "target_state_dict" in checkpoint:
                    self.target_net.load_state_dict(checkpoint["target_state_dict"])
                else:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.training_steps = int(checkpoint.get("training_steps", 0))
                self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
                self.policy_net.eval()
                self.target_net.eval()
                return
            except Exception:
                pass
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()

    def _save_dqn(self) -> None:
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "state_dim": self.STATE_DIM,
            "actions": self.ACTIONS,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint, self.dqn_model_path)
        torch.save(self.target_net.state_dict(), self.target_model_path)

    # ------------------------------------------------------------------
    # Replay memory
    # ------------------------------------------------------------------
    def _init_replay_db(self) -> None:
        """Create the SQLite replay-memory table used as the primary DQN memory."""
        try:
            with sqlite3.connect(self.replay_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS risk_dqn_replay (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at_utc TEXT,
                        state_text TEXT,
                        state_vector_json TEXT,
                        action TEXT,
                        action_index INTEGER,
                        reward REAL,
                        next_state_text TEXT,
                        next_state_vector_json TEXT,
                        done INTEGER,
                        source TEXT DEFAULT 'risk_agent'
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_dqn_replay_time ON risk_dqn_replay(created_at_utc)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_dqn_replay_action ON risk_dqn_replay(action)")
        except Exception:
            # CSV replay remains as a compatibility fallback.
            pass

    def _read_replay_from_sqlite(self) -> pd.DataFrame:
        try:
            if not self.replay_db_path.exists() or self.replay_db_path.stat().st_size == 0:
                return self._empty_replay_df()
            with sqlite3.connect(self.replay_db_path) as conn:
                table = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='risk_dqn_replay'"
                ).fetchone()
                if table is None:
                    return self._empty_replay_df()
                df = pd.read_sql_query(
                    """
                    SELECT created_at_utc, state_text, state_vector_json,
                           action, action_index, reward, next_state_text,
                           next_state_vector_json, done
                    FROM risk_dqn_replay
                    ORDER BY id ASC
                    """,
                    conn,
                )
            for col in self._empty_replay_df().columns:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception:
            return self._empty_replay_df()

    def _append_replay_to_sqlite(self, row: Dict[str, Any]) -> bool:
        try:
            self._init_replay_db()
            with sqlite3.connect(self.replay_db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO risk_dqn_replay (
                        created_at_utc, state_text, state_vector_json,
                        action, action_index, reward, next_state_text,
                        next_state_vector_json, done, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("created_at_utc"),
                        row.get("state_text"),
                        row.get("state_vector_json"),
                        row.get("action"),
                        int(row.get("action_index", 0)),
                        float(row.get("reward", 0.0)),
                        row.get("next_state_text"),
                        row.get("next_state_vector_json"),
                        1 if row.get("done") else 0,
                        "risk_agent",
                    ),
                )
            return True
        except Exception:
            return False

    def _empty_replay_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "created_at_utc",
                "state_text",
                "state_vector_json",
                "action",
                "action_index",
                "reward",
                "next_state_text",
                "next_state_vector_json",
                "done",
            ]
        )

    def _read_replay(self) -> pd.DataFrame:
        """
        Read replay memory from SQLite first, then fall back to CSV.
        CSV is kept as a mirror for compatibility with old dashboards.
        """
        sqlite_df = self._read_replay_from_sqlite()
        if not sqlite_df.empty:
            return sqlite_df

        if not self.replay_path.exists() or self.replay_path.stat().st_size == 0:
            return self._empty_replay_df()
        try:
            df = pd.read_csv(self.replay_path)
            for col in self._empty_replay_df().columns:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception:
            return self._empty_replay_df()

    def _append_replay(
        self,
        state_text: str,
        state_vector: List[float],
        action: str,
        reward: float,
        next_state_text: Optional[str] = None,
        next_state_vector: Optional[List[float]] = None,
        done: bool = True,
    ) -> int:
        action = action if action in self.ACTIONS else "KEEP_SIGNAL"
        next_state_vector = next_state_vector if next_state_vector is not None else state_vector
        row = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "state_text": state_text,
            "state_vector_json": json.dumps([float(x) for x in state_vector]),
            "action": action,
            "action_index": self.ACTION_INDEX[action],
            "reward": float(reward),
            "next_state_text": next_state_text or "",
            "next_state_vector_json": json.dumps([float(x) for x in next_state_vector]),
            "done": bool(done),
        }

        # SQLite is the primary replay memory. CSV is only a compatibility mirror.
        self._append_replay_to_sqlite(row)
        df = self._read_replay()
        try:
            csv_df = df.copy()
            csv_df.to_csv(self.replay_path, index=False)
        except Exception:
            pass
        return len(df)

    def _sample_replay_batch(self, df: pd.DataFrame) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(df) < self.min_replay_samples:
            return None
        sample_size = min(self.batch_size, len(df))
        batch = df.sample(sample_size, random_state=random.randint(1, 1_000_000))

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _, row in batch.iterrows():
            try:
                states.append(json.loads(row["state_vector_json"]))
                actions.append(int(row["action_index"]))
                rewards.append(float(row["reward"]))
                next_states.append(json.loads(row["next_state_vector_json"]))
                dones.append(1.0 if str(row.get("done", "True")).lower() in ["true", "1", "yes"] else 0.0)
            except Exception:
                continue

        if not states:
            return None

        return (
            torch.tensor(states, dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1),
        )

    # ------------------------------------------------------------------
    # DQN prediction and training
    # ------------------------------------------------------------------
    # No heuristic Q-value fallback is used. All advisory Q-values come from the PyTorch policy network.

    def _predict_q_values_from_vector(self, vector: List[float]) -> Dict[str, float]:
        """
        Strict DQN inference.

        Q-values are always produced by the PyTorch policy network.
        We do not use hand-written heuristic Q-values during warm-up.
        If replay memory is still small, the network is simply untrained/warm-up,
        and the hard safety layer remains the final guardrail.
        """
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor([vector], dtype=torch.float32, device=self.device)
            q_values = self.policy_net(state_tensor).detach().cpu().numpy()[0].tolist()
        return {action: float(q_values[idx]) for action, idx in self.ACTION_INDEX.items()}

    def _predict_q_values(
        self,
        vector: List[float],
        validation_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        signal_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Keep unused compatibility parameters so old app.py calls still work.
        return self._predict_q_values_from_vector(vector)

    def _choose_dqn_action(self, q_values: Dict[str, float]) -> str:
        if random.random() < self.epsilon:
            # Do not randomly explore BLOCK_TRADE in live paper decision mode.
            return random.choice(["KEEP_SIGNAL", "DOWNGRADE_TO_HOLD"])
        return max(q_values, key=q_values.get)

    def _train_dqn_from_replay(self) -> Dict[str, Any]:
        df = self._read_replay()
        replay_count = len(df)
        if replay_count < self.min_replay_samples:
            return {
                "success": False,
                "trained": False,
                "replay_count": replay_count,
                "min_replay_samples": self.min_replay_samples,
                "summary": f"DQN not trained yet. Need at least {self.min_replay_samples} replay samples.",
            }

        losses = []
        self.policy_net.train()
        self.target_net.eval()

        for _ in range(self.train_epochs_per_update):
            batch = self._sample_replay_batch(df)
            if batch is None:
                continue
            states, actions, rewards, next_states, dones = batch

            q_values = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
                target_q = rewards + self.gamma * next_q_values * (1.0 - dones)

            loss = self.loss_fn(q_values, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            self.training_steps += 1

            if self.training_steps % self.target_update_steps == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.policy_net.eval()
        self.target_net.eval()
        self._save_dqn()

        avg_loss = round(sum(losses) / len(losses), 6) if losses else None
        self.last_training_loss = avg_loss
        self._record_training_loss(avg_loss, replay_count)

        return {
            "success": True,
            "trained": True,
            "replay_count": replay_count,
            "training_steps": self.training_steps,
            "target_update_steps": self.target_update_steps,
            "epsilon": round(self.epsilon, 5),
            "loss": avg_loss,
            "recent_loss": avg_loss,
            "summary": "DQN policy network updated from replay memory.",
        }


    def _record_training_loss(self, loss_value: Optional[float], replay_count: int) -> None:
        """Keep a lightweight local loss log for diagnostics."""
        if loss_value is None:
            return
        row = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "loss": float(loss_value),
            "replay_count": int(replay_count),
            "training_steps": int(self.training_steps),
            "epsilon": float(self.epsilon),
        }
        try:
            df = pd.DataFrame([row])
            if self.loss_history_path.exists() and self.loss_history_path.stat().st_size > 0:
                old_df = pd.read_csv(self.loss_history_path)
                df = pd.concat([old_df, df], ignore_index=True).tail(1000)
            df.to_csv(self.loss_history_path, index=False)
        except Exception:
            pass

    def _recent_loss(self) -> Optional[float]:
        if self.last_training_loss is not None:
            return float(self.last_training_loss)
        try:
            if not self.loss_history_path.exists() or self.loss_history_path.stat().st_size == 0:
                return None
            df = pd.read_csv(self.loss_history_path)
            if df.empty or "loss" not in df.columns:
                return None
            return float(df["loss"].dropna().iloc[-1])
        except Exception:
            return None

    def _record_state_q_values(self, state: str, q_values: Dict[str, float]) -> None:
        self.q_table[state] = {action: float(q_values.get(action, 0.0)) for action in self.ACTIONS}
        self._save_q_table_compat()

    # ------------------------------------------------------------------
    # Risk decision rules
    # ------------------------------------------------------------------
    def _hard_safety_action(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        model_signal = self._model_signal(signal_result)
        validation_score = self._validation_score(validation_result)
        validation_action = self._validation_action(validation_result)
        model_conf = self._model_confidence(signal_result)
        entry_risk = self._entry_risk(analysis_result, signal_result)
        volatility = self._volatility_level(analysis_result)

        reasons = []
        if validation_action == "BLOCK_ANALYSIS":
            reasons.append("data validation blocked the analysis")
            return "BLOCK_TRADE", reasons
        if validation_score < 0.35:
            reasons.append("data confidence is too weak")
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

    def _filter_dqn_action(self, dqn_action: str, hard_action: str, model_signal: str) -> str:
        if hard_action == "BLOCK_TRADE":
            return "BLOCK_TRADE"
        if dqn_action == "BLOCK_TRADE":
            # DQN cannot hard-block by itself in the paper decision-support system.
            return "DOWNGRADE_TO_HOLD" if model_signal == "BUY_CANDIDATE" else "KEEP_SIGNAL"
        if dqn_action == "DOWNGRADE_TO_HOLD" and model_signal != "BUY_CANDIDATE":
            return "KEEP_SIGNAL"
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

    def _risk_level(
        self,
        final_signal: str,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        action: str,
    ) -> Tuple[str, str]:
        if action == "BLOCK_TRADE":
            return "Critical", "Data quality or safety rules blocked the result."

        points = 0
        notes = []
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
    def assess_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        validation_result = validation_result or {}
        analysis_result = analysis_result or {}
        signal_result = signal_result or {}

        symbol = self._get_symbol(signal_result, analysis_result, validation_result)
        model_signal = self._model_signal(signal_result)
        vector = self._state_vector(validation_result, analysis_result, signal_result)
        state = self._state_string(symbol, validation_result, analysis_result, signal_result)

        hard_action, hard_reasons = self._hard_safety_action(validation_result, analysis_result, signal_result)
        q_values = self._predict_q_values(vector, validation_result, analysis_result, signal_result)
        raw_dqn_action = self._choose_dqn_action(q_values)
        filtered_dqn_action = self._filter_dqn_action(raw_dqn_action, hard_action, model_signal)
        final_action = self._combine_actions(hard_action, filtered_dqn_action)
        final_signal = self._apply_action(model_signal, final_action)
        risk_level, risk_interpretation = self._risk_level(final_signal, validation_result, analysis_result, signal_result, final_action)
        self._record_state_q_values(state, q_values)

        replay_count = len(self._read_replay())
        dqn_status = "active" if replay_count >= self.min_replay_samples else "warmup"

        reasoning_steps = [
            f"Model signal: {model_signal}.",
            f"Validation confidence: {self._validation_confidence(validation_result)}.",
            f"Analyst signal: {self._analyst_signal(analysis_result)}.",
            f"Trend direction: {self._trend_direction(analysis_result, signal_result)}.",
            f"Entry timing risk: {self._entry_risk(analysis_result, signal_result)}.",
            f"Hard safety action: {hard_action} ({'; '.join(hard_reasons)}).",
            f"DQN status: {dqn_status}; replay samples: {replay_count}/{self.min_replay_samples}.",
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
            "agent_goal": "Apply hard safety checks and strict DQN risk advisory.",
            "symbol": symbol,
            "original_signal": model_signal,
            "risk_action": final_action,
            "hard_safety_action": hard_action,
            "dqn_action": raw_dqn_action,
            "filtered_dqn_action": filtered_dqn_action,
            "dqn_q_values": {k: round(v, 5) for k, v in q_values.items()},
            "dqn_status": dqn_status,
            "dqn_framework": "PyTorch DQN with replay memory and target network",
            "dqn_model_path": str(self.dqn_model_path),
            "dqn_target_model_path": str(self.target_model_path),
            "dqn_replay_path": str(self.replay_path),
            "dqn_replay_db_path": str(self.replay_db_path),
            "dqn_replay_storage": "sqlite_primary_csv_mirror",
            "dqn_replay_count": replay_count,
            "dqn_min_replay_samples": self.min_replay_samples,
            "epsilon": round(self.epsilon, 5),
            "recent_dqn_loss": self._recent_loss(),
            "dqn_training_ready": replay_count >= self.min_replay_samples,
            "dqn_diagnostics": {
                "q_values": {k: round(v, 5) for k, v in q_values.items()},
                "epsilon": round(self.epsilon, 5),
                "replay_count": replay_count,
                "min_replay_samples": self.min_replay_samples,
                "recent_loss": self._recent_loss(),
                "training_steps": self.training_steps,
                "target_update_steps": self.target_update_steps,
                "status": dqn_status,
            },
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
                "dqn_status": dqn_status,
                "risk_interpretation": risk_interpretation,
                "explanation_for_llm": f"Risk level is {risk_level}. {risk_interpretation}",
            },
            "summary": f"Risk Agent set final signal to {final_signal} with {risk_level} risk.",
        }

    # Aliases expected by app.py
    def apply_risk_control(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    def adjust_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    def evaluate_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    def control_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    def run(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

    # ------------------------------------------------------------------
    # Delayed reward learning
    # ------------------------------------------------------------------
    def calculate_reward(self, final_signal: str, future_return: float, volatility_level: str = "Unknown") -> float:
        future_return = self._safe_float(future_return, 0.0) or 0.0
        final_signal = self._normalise_label(final_signal, "HOLD")

        if final_signal == "BUY_CANDIDATE":
            reward = future_return
        elif final_signal == "SELL_RISK":
            reward = -future_return
        elif final_signal == "HOLD":
            reward = -abs(future_return) * 0.15
        elif final_signal == "BLOCKED":
            reward = abs(future_return) * 0.30 if future_return < 0 else -future_return * 0.20
        else:
            reward = 0.0

        if str(volatility_level).title() in ["High", "Critical"]:
            reward -= 0.003
        return float(reward)

    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compatibility name for RewardAgent.
        Under strict DQN, this appends a transition to replay memory and trains
        the policy network only when enough replay samples exist.
        """
        if not state:
            return {"success": False, "summary": "Cannot update DQN because state is missing."}

        action = action if action in self.ACTIONS else "KEEP_SIGNAL"
        reward = self._safe_float(reward, 0.0) or 0.0
        state_vector = self._parse_state_string_to_vector(state)
        next_state_vector = self._parse_state_string_to_vector(next_state) if next_state else state_vector

        old_q = self._predict_q_values_from_vector(state_vector).get(action, 0.0)
        replay_count = self._append_replay(
            state_text=state,
            state_vector=state_vector,
            action=action,
            reward=reward,
            next_state_text=next_state,
            next_state_vector=next_state_vector,
            done=next_state is None,
        )
        train_result = self._train_dqn_from_replay()
        new_q = self._predict_q_values_from_vector(state_vector).get(action, old_q)
        q_values = self._predict_q_values(state_vector)
        self._record_state_q_values(state, q_values)

        return {
            "success": True,
            "learning_type": "strict_pytorch_dqn_replay_update",
            "state": state,
            "action": action,
            "reward": round(reward, 6),
            "old_q": round(old_q, 6),
            "new_q": round(new_q, 6),
            "replay_count": replay_count,
            "min_replay_samples": self.min_replay_samples,
            "dqn_model_path": str(self.dqn_model_path),
            "dqn_target_model_path": str(self.target_model_path),
            "dqn_replay_path": str(self.replay_path),
            "dqn_replay_db_path": str(self.replay_db_path),
            "dqn_replay_storage": "sqlite_primary_csv_mirror",
            "train_result": train_result,
            "summary": "Added feedback to DQN replay memory and trained the policy network when enough samples were available.",
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
        result.update(
            {
                "final_signal": final_signal,
                "risk_level": risk_level,
                "future_return": self._safe_float(future_return, 0.0),
                "calculated_reward": round(reward, 6),
            }
        )
        return result
