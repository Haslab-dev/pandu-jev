"""Pandu Chess Policy Architecture (~43.9K parameters) with Factorized Legal-Move Masking.

Decouples rule verification from neural learning:
1. Backbone extracts 96-dim policy latent from 224-dim ChessFeatureEncoder.
2. Factorized output heads score:
   - from_head: 64 squares
   - to_head: 64 squares
   - promo_head: 5 promotion classes (None, Q, R, B, N)
   - value_head: 1 position evaluation scalar
3. Legal Move Masking guarantees 100% legal moves:
   Only legal moves in the current position are scored and passed through softmax.
"""

from typing import Dict, List, Optional, Tuple
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arena.chess_encoder import ChessFeatureEncoder

PROMO_MAP: Dict[Optional[chess.PieceType], int] = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}

REV_PROMO_MAP = {v: k for k, v in PROMO_MAP.items()}

CHESS_INTENTS: Dict[int, str] = {
    0: "DEVELOPMENT",
    1: "WIN_MATERIAL",
    2: "DEFENSE",
    3: "MATE_ATTACK",
    4: "CENTER_CONTROL",
    5: "ENDGAME_PUSH",
    6: "TACTICAL_COMBINATION",
    7: "PIECE_IMPROVEMENT",
}

NUM_INTENTS = len(CHESS_INTENTS)


class PolicyDecision(tuple):
    """3-tuple compatible with (move, conf, val) plus intent attribute."""

    def __new__(cls, move: chess.Move, confidence: float, value: float, intent: str = "DEVELOPMENT"):
        obj = super().__new__(cls, (move, confidence, value))
        obj.move = move
        obj.confidence = confidence
        obj.value = value
        obj.intent = intent
        return obj


class PanduChessNet(nn.Module):
    """Neural policy network for Pandu Chess with factorized heads (~44.7K parameters)."""

    def __init__(self, in_dim: int = 224, hidden_dim: int = 96):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Factorized action heads
        self.from_head = nn.Linear(hidden_dim, 64)
        self.to_head = nn.Linear(hidden_dim, 64)
        self.promo_head = nn.Linear(hidden_dim, 5)

        # Auxiliary heads
        self.value_head = nn.Linear(hidden_dim, 1)
        self.intent_head = nn.Linear(hidden_dim, NUM_INTENTS)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (from_logits, to_logits, promo_logits, value, intent_logits)."""
        latent = self.backbone(x)
        from_logits = self.from_head(latent)
        to_logits = self.to_head(latent)
        promo_logits = self.promo_head(latent)
        value = torch.tanh(self.value_head(latent))
        intent_logits = self.intent_head(latent)
        return from_logits, to_logits, promo_logits, value, intent_logits

    def score_legal_moves(
        self, x: torch.Tensor, legal_moves: List[chess.Move]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute unmasked logits and softmax probabilities strictly over legal moves.

        Returns:
            legal_logits: 1D Tensor of length len(legal_moves)
            value: scalar Tensor
            intent_logits: 1D Tensor of length NUM_INTENTS
        """
        from_logits, to_logits, promo_logits, value, intent_logits = self.forward(x)
        if x.dim() > 1:
            from_logits = from_logits[0]
            to_logits = to_logits[0]
            promo_logits = promo_logits[0]
            value = value[0]
            intent_logits = intent_logits[0]

        scores = []
        for m in legal_moves:
            p_idx = PROMO_MAP.get(m.promotion, 0)
            move_score = from_logits[m.from_square] + to_logits[m.to_square] + promo_logits[p_idx]
            scores.append(move_score)

        if not scores:
            return torch.empty(0, device=x.device), value, intent_logits

        legal_logits = torch.stack(scores)
        return legal_logits, value, intent_logits


class PanduChessPolicy:
    """High-level wrapper for Pandu Chess inference and move selection with legal-move masking."""

    def __init__(
        self,
        model: Optional[PanduChessNet] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model if model is not None else PanduChessNet()
        self.model.to(self.device)
        self.encoder = ChessFeatureEncoder()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def select_move(
        self,
        board: chess.Board,
        temperature: float = 1.0,
        deterministic: bool = True,
    ) -> PolicyDecision:
        """Select a guaranteed legal move using legal-move masked softmax.

        Returns:
            PolicyDecision(move, confidence, estimated_value, intent)
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return PolicyDecision(chess.Move.null(), 0.0, 0.0, "DEFENSE")

        if len(legal_moves) == 1:
            return PolicyDecision(legal_moves[0], 1.0, 0.0, "FORCED")

        # 1. Invariant Rule Guard: 100% Checkmate Recognition
        for m in legal_moves:
            board.push(m)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return PolicyDecision(m, 1.0, 1.0, "MATE_ATTACK")

        feat = self.encoder.encode(board, perspective=board.turn)
        x = torch.tensor(feat, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            legal_logits, value, intent_logits = self.model.score_legal_moves(x, legal_moves)

            # Avoid immediate stalemate/insufficient material collapsing if winning
            for i, m in enumerate(legal_moves):
                board.push(m)
                if board.is_stalemate() or board.is_insufficient_material():
                    legal_logits[i] -= 20.0
                elif board.is_repetition(2):
                    legal_logits[i] -= 3.0
                board.pop()

            intent_idx = int(torch.argmax(intent_logits).item())
            intent_name = CHESS_INTENTS.get(intent_idx, "DEVELOPMENT")

            if temperature <= 0.01 or deterministic:
                best_idx = int(torch.argmax(legal_logits).item())
                probs = F.softmax(legal_logits, dim=0)
                conf = float(probs[best_idx].item())
                return PolicyDecision(legal_moves[best_idx], conf, float(value.item()), intent_name)
            else:
                scaled_logits = legal_logits / temperature
                probs = F.softmax(scaled_logits, dim=0)
                sampled_idx = int(torch.multinomial(probs, 1).item())
                conf = float(probs[sampled_idx].item())
                return PolicyDecision(legal_moves[sampled_idx], conf, float(value.item()), intent_name)
