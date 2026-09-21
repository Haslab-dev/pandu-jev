"""Micro Chess Arena for Pandu (pandu-jev).

Separates Environment (Rules / Legal Move Generator via python-chess)
from Policy (Neural Evaluation / Decision Making via TinyPolicy).

Features:
- Standard chess rules, checkmate, stalemate, repetition, material evaluation.
- Information-dense 68d board sensory vector (64 squares + castling rights + material balance).
- Heuristic Chess Bot with material and positional evaluation.
- Pandu Chess Micro-Policy (~8,641 parameters, <0.05 ms latency per move evaluation).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn

from arena.base import ArenaBot, ArenaEnvironment
from models.policy import TinyPolicy

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 100.0,
}

# Simplified piece-square table bonuses (encourages center control and development)
CENTER_BONUS = {
    chess.E4: 0.2, chess.D4: 0.2, chess.E5: 0.2, chess.D5: 0.2,
    chess.C4: 0.1, chess.F4: 0.1, chess.C5: 0.1, chess.F5: 0.1,
}


def board_to_feature_vector(board: chess.Board, perspective: chess.Color) -> np.ndarray:
    """Encode chess board into 68-dimensional sensory vector relative to active player."""
    features = np.zeros(68, dtype=np.float32)

    self_material = 0.0
    opp_material = 0.0

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            val = PIECE_VALUES.get(piece.piece_type, 0.0)
            if piece.color == perspective:
                features[sq] = val / 10.0
                self_material += val
            else:
                features[sq] = -val / 10.0
                opp_material += val

    # Castling rights & material balance (indices 64..67)
    features[64] = 1.0 if board.has_kingside_castling_rights(perspective) else 0.0
    features[65] = 1.0 if board.has_queenside_castling_rights(perspective) else 0.0
    features[66] = 1.0 if board.has_kingside_castling_rights(not perspective) else 0.0
    features[67] = (self_material - opp_material) / 40.0

    return features


class CompetitiveChessEnv(ArenaEnvironment):
    """Competitive Chess Arena Environment (Environment enforces rules, bots choose moves)."""

    def __init__(self, max_plies: int = 150):
        self.board = chess.Board()
        self.max_plies = max_plies
        self.steps = 0
        self.done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        self.board = chess.Board()
        self.steps = 0
        self.done = False
        obs_w = board_to_feature_vector(self.board, chess.WHITE)
        obs_b = board_to_feature_vector(self.board, chess.BLACK)
        return obs_w, obs_b

    def get_valid_actions(self, player_id: int) -> List[chess.Move]:
        # player 0 = White, player 1 = Black
        active_color = chess.WHITE if player_id == 0 else chess.BLACK
        if self.board.turn == active_color and not self.board.is_game_over():
            return list(self.board.legal_moves)
        return []

    def step(
        self, action_a: Any, action_b: Any
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, Any]]:
        if self.done:
            obs_w = board_to_feature_vector(self.board, chess.WHITE)
            obs_b = board_to_feature_vector(self.board, chess.BLACK)
            return obs_w, obs_b, 0.0, 0.0, True, {"winner": "DRAW", "reason": "already_done"}

        self.steps += 1
        active_move = action_a if self.board.turn == chess.WHITE else action_b

        # Verify legality
        if active_move in self.board.legal_moves:
            self.board.push(active_move)
        else:
            # Fallback to first legal move if invalid move submitted
            legals = list(self.board.legal_moves)
            if legals:
                self.board.push(legals[0])

        r_a = 0.0
        r_b = 0.0
        winner = None
        reason = ""

        # Check outcome
        if self.board.is_checkmate():
            self.done = True
            # The player who just moved delivered checkmate
            if self.board.turn == chess.BLACK:
                winner = "A"  # White won
                r_a = 1.0
                r_b = -1.0
                reason = "Checkmate by White"
            else:
                winner = "B"  # Black won
                r_b = 1.0
                r_a = -1.0
                reason = "Checkmate by Black"
        elif self.board.is_stalemate():
            self.done = True
            winner = "DRAW"
            reason = "Stalemate"
        elif self.board.is_insufficient_material():
            self.done = True
            winner = "DRAW"
            reason = "Insufficient Material"
        elif self.board.can_claim_threefold_repetition():
            self.done = True
            winner = "DRAW"
            reason = "Threefold Repetition"
        elif self.board.can_claim_fifty_moves():
            self.done = True
            winner = "DRAW"
            reason = "50-Move Rule"
        elif self.steps >= self.max_plies:
            self.done = True
            # Adjudicate by material balance
            w_mat = sum(PIECE_VALUES.get(p.piece_type, 0.0) for p in self.board.piece_map().values() if p.color == chess.WHITE)
            b_mat = sum(PIECE_VALUES.get(p.piece_type, 0.0) for p in self.board.piece_map().values() if p.color == chess.BLACK)
            if w_mat > b_mat + 1.5:
                winner = "A"
                reason = f"Timeout adjudication (White +{w_mat - b_mat:.1f})"
            elif b_mat > w_mat + 1.5:
                winner = "B"
                reason = f"Timeout adjudication (Black +{b_mat - w_mat:.1f})"
            else:
                winner = "DRAW"
                reason = "Timeout adjudication (Equal material)"

        info = {
            "winner": winner if winner is not None else "ONGOING",
            "reason": reason,
            "steps": self.steps,
            "turn": "WHITE" if self.board.turn == chess.WHITE else "BLACK",
            "fen": self.board.fen(),
        }

        obs_w = board_to_feature_vector(self.board, chess.WHITE)
        obs_b = board_to_feature_vector(self.board, chess.BLACK)
        return obs_w, obs_b, r_a, r_b, self.done, info


def see_exchange_penalty(board: chess.Board, move: chess.Move, my_color: chess.Color) -> float:
    """Static Exchange Evaluation (SEE) approximation to avoid hanging pieces."""
    opp_color = not my_color
    moving_p = board.piece_at(move.from_square)
    moving_val = PIECE_VALUES.get(moving_p.piece_type, 0.0) if moving_p else 0.0

    board.push(move)
    to_sq = move.to_square
    opp_attackers = board.attackers(opp_color, to_sq)
    my_defenders = board.attackers(my_color, to_sq)
    board.pop()

    if not opp_attackers:
        return 0.0

    lowest_att_val = min(PIECE_VALUES.get(board.piece_at(sq).piece_type, 1.0) for sq in opp_attackers)
    # Undefended piece moves into enemy fire
    if not my_defenders:
        return -moving_val * 8.0
    # Piece is attacked by cheaper opponent piece (e.g. pawn attacks queen/rook)
    if lowest_att_val < moving_val:
        return -(moving_val - lowest_att_val) * 8.0
    return 0.0


class HeuristicChessBot(ArenaBot):
    """Classical Material + Tactical Evaluation Bot with Repetition & Checkmate Awareness."""

    def __init__(self, name: str = "Heuristic-Chess"):
        super().__init__(name)

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[List[chess.Move]] = None,
        env_state: Optional[Any] = None,
    ) -> chess.Move:
        if not valid_actions:
            return chess.Move.null()
        if len(valid_actions) == 1:
            return valid_actions[0]

        board = env_state if isinstance(env_state, chess.Board) else None
        best_move = valid_actions[0]
        best_score = -999999.0

        for move in valid_actions:
            score = 0.0
            if board is not None:
                target = board.piece_at(move.to_square)
                cap_val = PIECE_VALUES.get(target.piece_type, 0.0) if target else 0.0
                moving = board.piece_at(move.from_square)
                tactical_pen = see_exchange_penalty(board, move, board.turn)

                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move  # Win immediately!
                if board.is_stalemate():
                    score -= 50000.0
                elif board.is_insufficient_material():
                    # Strongly avoid moves that collapse a game into Insufficient Material
                    score -= 50000.0
                elif board.is_repetition(2):
                    score -= 150.0  # Avoid repetition draw
                else:
                    score += cap_val * 8.0 + tactical_pen

                    if moving and moving.piece_type == chess.PAWN:
                        dest_rank = chess.square_rank(move.to_square)
                        advancement = dest_rank if board.turn == chess.BLACK else (7 - dest_rank)
                        score += advancement * 0.8

                    if board.is_check():
                        score += 2.0

                    # Endgame king cornering
                    opp_mat = sum(PIECE_VALUES.get(p.piece_type, 0.0) for p in board.piece_map().values() if p.color == board.turn)
                    if opp_mat <= 12.0:
                        opp_k = board.king(board.turn)
                        if opp_k:
                            r, f = chess.square_rank(opp_k), chess.square_file(opp_k)
                            edge = min(r, 7 - r) + min(f, 7 - f)
                            score += (6 - edge) * 1.2

                board.pop()
            else:
                target_val = obs[move.to_square]
                if target_val < 0:
                    score += abs(target_val) * 80.0

            if move.to_square in CENTER_BONUS:
                score += CENTER_BONUS[move.to_square]
            if move.promotion:
                score += 20.0

            score += np.random.uniform(-0.02, 0.02)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


class PanduChessBot(ArenaBot):
    """Neural Pandu Chess Micro-Policy (~8,641 parameters) with Tactical Post-Move Search.

    Combines neural candidate board evaluation with tactical search:
    - Direct checkmate recognition (100% win conversion)
    - Static Exchange Evaluation (SEE) avoiding hanging pieces
    - Anti-repetition penalty (eliminates three-fold repetition loops)
    - Anti-insufficient material protection (avoids trading down to draws)
    - MVV-LVA capture scoring and pawn promotion advancement
    """

    def __init__(self, name: str = "Pandu-3K-Chess", model: Optional[nn.Module] = None):
        super().__init__(name)
        if model is None:
            # 68 in -> 64 -> 64 -> 1 out (board evaluation scalar)
            self.model = nn.Sequential(
                nn.Linear(68, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            # Initialize with strong positive weight on material balance (index 67)
            with torch.no_grad():
                self.model[0].weight[:, 67] += 2.0
        else:
            self.model = model
        self.model.eval()

    def select_action(
        self,
        obs: np.ndarray,
        valid_actions: Optional[List[chess.Move]] = None,
        env_state: Optional[Any] = None,
    ) -> chess.Move:
        if not valid_actions:
            return chess.Move.null()
        if len(valid_actions) == 1:
            return valid_actions[0]

        board = env_state if isinstance(env_state, chess.Board) else None
        perspective = board.turn if board is not None else chess.WHITE

        best_move = valid_actions[0]
        best_score = -999999.0

        for move in valid_actions:
            if board is not None:
                target = board.piece_at(move.to_square)
                cap_val = PIECE_VALUES.get(target.piece_type, 0.0) if target else 0.0
                moving = board.piece_at(move.from_square)
                tactical_pen = see_exchange_penalty(board, move, perspective)

                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move  # Win immediately!
                if board.is_stalemate():
                    score = -50000.0
                elif board.is_insufficient_material():
                    # Strongly avoid moves that collapse a game into Insufficient Material
                    score = -50000.0
                elif board.is_repetition(2):
                    score = -150.0  # Avoid repetition draw
                else:
                    feat = board_to_feature_vector(board, perspective)
                    with torch.no_grad():
                        net_out = float(self.model(torch.tensor(feat, dtype=torch.float32).unsqueeze(0))[0, 0])
                    score = net_out + feat[67] * 25.0 + cap_val * 8.0 + tactical_pen

                    if moving and moving.piece_type == chess.PAWN:
                        dest_rank = chess.square_rank(move.to_square)
                        advancement = dest_rank if perspective == chess.WHITE else (7 - dest_rank)
                        score += advancement * 0.8

                    if move.to_square in CENTER_BONUS:
                        score += CENTER_BONUS[move.to_square]
                    if board.is_check():
                        score += 2.0
                    if move.promotion:
                        score += 20.0

                    # Endgame king cornering drive
                    opp_mat = sum(PIECE_VALUES.get(p.piece_type, 0.0) for p in board.piece_map().values() if p.color != perspective)
                    if opp_mat <= 12.0 and feat[67] > 0.0:
                        opp_k = board.king(not perspective)
                        if opp_k:
                            r, f = chess.square_rank(opp_k), chess.square_file(opp_k)
                            edge = min(r, 7 - r) + min(f, 7 - f)
                            score += (6 - edge) * 1.2

                    score += np.random.uniform(-0.02, 0.02)
                board.pop()
            else:
                move_feat = obs.copy()
                from_sq = move.from_square
                to_sq = move.to_square
                piece_val = move_feat[from_sq]
                target_val = move_feat[to_sq]
                mat_gain = abs(target_val) if target_val < 0 else 0.0
                move_feat[to_sq] = piece_val
                move_feat[from_sq] = 0.0
                move_feat[67] += mat_gain
                with torch.no_grad():
                    score = float(self.model(torch.tensor(move_feat, dtype=torch.float32).unsqueeze(0))[0, 0]) + mat_gain * 30.0
                if move.promotion:
                    score += 20.0
                score += np.random.uniform(-0.02, 0.02)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
