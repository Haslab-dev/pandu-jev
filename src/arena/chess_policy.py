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

import os
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
    """Neural policy network for Pandu Chess with factorized heads (~44.7K - ~85K parameters)."""

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

    def load_checkpoint(self, path: str) -> None:
        """Load weights from file, dynamically adjusting hidden_dim if necessary."""
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location=self.device)
        if "from_head.weight" in state:
            h_dim = state["from_head.weight"].shape[1]
            if h_dim != self.model.hidden_dim:
                self.model = PanduChessNet(hidden_dim=h_dim).to(self.device)
        self.model.load_state_dict(state)

    def select_move(
        self,
        board: chess.Board,
        temperature: float = 1.0,
        deterministic: bool = True,
        use_tactical_guard: bool = True,
    ) -> PolicyDecision:
        """Select a guaranteed legal move using legal-move masked softmax and tactical guard.

        Returns:
            PolicyDecision(move, confidence, estimated_value, intent)
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return PolicyDecision(chess.Move.null(), 0.0, 0.0, "DEFENSE")

        if len(legal_moves) == 1:
            return PolicyDecision(legal_moves[0], 1.0, 0.0, "FORCED")

        # 1. Canonical Perspective Mirroring:
        # When Black plays, mirror the board vertically so the policy network ALWAYS evaluates
        # from White's perspective. This completely cures asymmetric opening blunders as Black.
        is_black = (board.turn == chess.BLACK)
        eval_board = board.mirror() if is_black else board
        eval_legals = list(eval_board.legal_moves)

        # 2. Invariant Rule Guard: 100% Immediate Checkmate Recognition
        for m in eval_legals:
            eval_board.push(m)
            is_mate = eval_board.is_checkmate()
            eval_board.pop()
            if is_mate:
                real_m = chess.Move(
                    chess.square_mirror(m.from_square),
                    chess.square_mirror(m.to_square),
                    m.promotion,
                ) if is_black else m
                return PolicyDecision(real_m, 1.0, 1.0, "MATE_ATTACK")

        feat = self.encoder.encode(eval_board, perspective=chess.WHITE)
        x = torch.tensor(feat, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            legal_logits, value, intent_logits = self.model.score_legal_moves(x, eval_legals)

            intent_idx = int(torch.argmax(intent_logits).item())
            intent_name = CHESS_INTENTS.get(intent_idx, "DEVELOPMENT")

            if use_tactical_guard:
                from arena.chess import CENTER_BONUS, PIECE_VALUES, see_exchange_penalty

                def _get_quiescence_loss(b: chess.Board) -> float:
                    max_loss = 0.0
                    for opp_m in b.legal_moves:
                        if b.is_capture(opp_m):
                            victim = b.piece_at(opp_m.to_square)
                            attacker = b.piece_at(opp_m.from_square)
                            if victim and attacker:
                                v_val = PIECE_VALUES.get(victim.piece_type, 0.0)
                                a_val = PIECE_VALUES.get(attacker.piece_type, 0.0)
                                p = see_exchange_penalty(b, opp_m, chess.BLACK)
                                if p >= 0:
                                    loss = v_val - (a_val if p < 0 else 0.0)
                                    if loss > max_loss:
                                        max_loss = loss
                    return max_loss

                scored_moves = []
                for i, m in enumerate(eval_legals):
                    pen = see_exchange_penalty(eval_board, m, chess.WHITE)
                    target = eval_board.piece_at(m.to_square)
                    cap_val = PIECE_VALUES.get(target.piece_type, 0.0) if target else 0.0
                    moving = eval_board.piece_at(m.from_square)

                    castling_bonus = 0.0
                    if eval_board.is_castling(m):
                        castling_bonus = 4.0
                    elif moving and moving.piece_type == chess.KING:
                        total_mat = sum(PIECE_VALUES.get(p.piece_type, 0.0) for p in eval_board.piece_map().values())
                        if total_mat > 35.0:
                            castling_bonus = -4.0  # King should not wander early in opening!

                    eval_board.push(m)
                    if eval_board.is_checkmate():
                        eval_board.pop()
                        real_m = chess.Move(
                            chess.square_mirror(m.from_square),
                            chess.square_mirror(m.to_square),
                            m.promotion,
                        ) if is_black else m
                        return PolicyDecision(real_m, 1.0, 1.0, "MATE_ATTACK")
                    if eval_board.is_stalemate() or eval_board.is_insufficient_material():
                        eval_board.pop()
                        scored_moves.append(-50000.0)
                        continue
                    if eval_board.is_repetition(2):
                        pen -= 150.0

                    hang_loss = _get_quiescence_loss(eval_board)

                    # Post-move material balance
                    w_mat = sum(PIECE_VALUES.get(p.piece_type, 0.0) for p in eval_board.piece_map().values() if p.color == chess.WHITE)
                    b_mat = sum(PIECE_VALUES.get(p.piece_type, 0.0) for p in eval_board.piece_map().values() if p.color == chess.BLACK)
                    mat_diff = w_mat - b_mat

                    pawn_adv = 0.0
                    if moving and moving.piece_type == chess.PAWN:
                        pawn_adv = chess.square_rank(m.to_square) * 0.8
                    promo = 20.0 if m.promotion else 0.0
                    center_b = CENTER_BONUS.get(m.to_square, 0.0)

                    corner_b = 0.0
                    if b_mat <= 12.0 and mat_diff > 1.5:
                        opp_k = eval_board.king(chess.BLACK)
                        if opp_k:
                            r, f = chess.square_rank(opp_k), chess.square_file(opp_k)
                            corner_b = (6 - min(r, 7 - r) - min(f, 7 - f)) * 1.5
                    eval_board.pop()

                    # Neural guidance + tactical exchange safety + 2-ply quiescence guard
                    combined_score = (
                        legal_logits[i].item() * 0.5
                        + mat_diff * 25.0
                        + cap_val * 8.0
                        + pen
                        - hang_loss * 25.0
                        + pawn_adv
                        + promo
                        + center_b
                        + castling_bonus
                        + corner_b
                        + np.random.uniform(-0.02, 0.02)
                    )
                    scored_moves.append(combined_score)

                best_idx = int(np.argmax(scored_moves))
                chosen_rel = eval_legals[best_idx]
                real_m = chess.Move(
                    chess.square_mirror(chosen_rel.from_square),
                    chess.square_mirror(chosen_rel.to_square),
                    chosen_rel.promotion,
                ) if is_black else chosen_rel

                probs = F.softmax(legal_logits, dim=0)
                conf = float(probs[best_idx].item())
                return PolicyDecision(real_m, conf, float(value.item()), intent_name)
            else:
                for i, m in enumerate(eval_legals):
                    eval_board.push(m)
                    if eval_board.is_stalemate() or eval_board.is_insufficient_material():
                        legal_logits[i] -= 20.0
                    elif eval_board.is_repetition(2):
                        legal_logits[i] -= 3.0
                    eval_board.pop()

                if temperature <= 0.01 or deterministic:
                    best_idx = int(torch.argmax(legal_logits).item())
                    probs = F.softmax(legal_logits, dim=0)
                    conf = float(probs[best_idx].item())
                    chosen_rel = eval_legals[best_idx]
                    real_m = chess.Move(
                        chess.square_mirror(chosen_rel.from_square),
                        chess.square_mirror(chosen_rel.to_square),
                        chosen_rel.promotion,
                    ) if is_black else chosen_rel
                    return PolicyDecision(real_m, conf, float(value.item()), intent_name)
                else:
                    scaled_logits = legal_logits / temperature
                    probs = F.softmax(scaled_logits, dim=0)
                    sampled_idx = int(torch.multinomial(probs, 1).item())
                    conf = float(probs[sampled_idx].item())
                    chosen_rel = eval_legals[sampled_idx]
                    real_m = chess.Move(
                        chess.square_mirror(chosen_rel.from_square),
                        chess.square_mirror(chosen_rel.to_square),
                        chosen_rel.promotion,
                    ) if is_black else chosen_rel
                    return PolicyDecision(real_m, conf, float(value.item()), intent_name)
