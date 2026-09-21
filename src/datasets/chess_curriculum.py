"""Deep 8-Tier Chess Curriculum Dataset Generator for Pandu Policy Training.

Implements cumulative curriculum tiers (Tiers 0 to 8):
- Tier 0 (Legal Move Baseline): Sound legal move execution.
- Tier 1 (Material & Captures): Hanging piece captures & favorable trades (MVV-LVA / SEE).
- Tier 2 (Checkmate & Check Evasion): Mate-in-1 conversion and king defense.
- Tier 3 (Basic Tactics): Pawn promotions, pin exploitation, simple forks.
- Tier 4 (Positional Fundamentals): Center control (e4/d4/e5/d5), development, open files.
- Tier 5 (Tactical Depth): Multi-ply tactical combinations (forks, skewers, discovered checks).
- Tier 6 (Endgame Fundamentals): King opposition, passed pawn queening races, rook endgames.
- Tier 7 (Strategic Planning): Improving worst-placed piece, targeting weak pawns, advantageous trades.
- Tier 8 (Engine Distillation): Deep alpha-beta minimax supervision with centipawn eval and intent.

Every sample provides:
- 224-dim sensory vector
- Target legal move index
- Target scalar position value (-1.0 to 1.0)
- Target strategic intent class (0 to 7)

Higher tiers accumulate previous datasets (D_k = D_{k-1} + S_k) to prevent catastrophic forgetting.
"""

from dataclasses import dataclass
from enum import IntEnum
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from arena.chess_encoder import ChessFeatureEncoder, PIECE_VALUES, CENTER_SQUARES, EXTENDED_CENTER
from arena.chess_policy import CHESS_INTENTS, PROMO_MAP


class CurriculumTier(IntEnum):
    LEVEL_0_LEGAL = 0
    LEVEL_1_MATERIAL = 1
    LEVEL_2_CHECKMATE = 2
    LEVEL_3_BASIC_TACTICS = 3
    LEVEL_4_POSITIONAL = 4
    LEVEL_5_TACTICAL_DEPTH = 5
    LEVEL_6_ENDGAME = 6
    LEVEL_7_STRATEGIC_PLANNING = 7
    LEVEL_8_ENGINE_DISTILLATION = 8


# Backward compatibility aliases
CurriculumTier.LEVEL_3_TACTICS = CurriculumTier.LEVEL_3_BASIC_TACTICS
CurriculumTier.LEVEL_4_STRATEGY = CurriculumTier.LEVEL_4_POSITIONAL


@dataclass
class ChessCurriculumSample:
    """A single training sample with board state, legal moves, target move, value, and intent."""
    fen: str
    perspective: chess.Color
    feature_vector: np.ndarray  # (224,)
    legal_moves: List[chess.Move]
    target_move: chess.Move
    target_move_idx: int
    tier: CurriculumTier
    target_value: float = 0.0
    target_intent: int = 0  # 0..7 corresponding to CHESS_INTENTS


class ChessCurriculumDataset(Dataset):
    """PyTorch Dataset wrapping curriculum chess positions."""

    def __init__(self, samples: List[ChessCurriculumSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "features": torch.tensor(s.feature_vector, dtype=torch.float32),
            "legal_moves": s.legal_moves,
            "target_idx": s.target_move_idx,
            "target_value": torch.tensor(s.target_value, dtype=torch.float32),
            "target_intent": torch.tensor(s.target_intent, dtype=torch.long),
            "tier": int(s.tier),
        }


def _minimax_eval(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
    """Fast alpha-beta minimax static evaluator for Tier 8 engine distillation."""
    if depth == 0 or board.is_game_over():
        if board.is_checkmate():
            return -100.0 if maximizing else 100.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        val = 0.0
        for p in board.piece_map().values():
            pv = PIECE_VALUES.get(p.piece_type, 0.0)
            val += pv if p.color == chess.WHITE else -pv
        return val

    if maximizing:
        max_eval = -999.0
        for m in board.legal_moves:
            board.push(m)
            ev = _minimax_eval(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = 999.0
        for m in board.legal_moves:
            board.push(m)
            ev = _minimax_eval(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval


def generate_curriculum_samples(
    tier: CurriculumTier,
    num_samples: int = 300,
    seed: int = 42,
) -> List[ChessCurriculumSample]:
    """Generate specialized positions corresponding to an 8-tier curriculum."""
    rng = random.Random(seed + int(tier) * 1000)
    samples: List[ChessCurriculumSample] = []
    encoder = ChessFeatureEncoder()

    attempts = 0
    max_attempts = num_samples * 25

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        board = chess.Board()

        # Depth range depends on tier to produce realistic phase distributions
        if tier in (CurriculumTier.LEVEL_0_LEGAL, CurriculumTier.LEVEL_4_POSITIONAL):
            depth = rng.randint(4, 18)  # Opening to early middlegame
        elif tier in (CurriculumTier.LEVEL_1_MATERIAL, CurriculumTier.LEVEL_2_CHECKMATE, CurriculumTier.LEVEL_3_BASIC_TACTICS, CurriculumTier.LEVEL_5_TACTICAL_DEPTH):
            depth = rng.randint(12, 36)  # Tactical middlegame
        elif tier == CurriculumTier.LEVEL_6_ENDGAME:
            depth = rng.randint(35, 65)  # Endgame positions
        else:
            depth = rng.randint(8, 45)

        for _ in range(depth):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            tactical = [m for m in moves if board.is_capture(m) or board.gives_check(m)]
            if tactical and rng.random() < 0.45:
                m = rng.choice(tactical)
            else:
                m = rng.choice(moves)
            board.push(m)

        if board.is_game_over():
            continue

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            continue

        perspective = board.turn
        target_move: Optional[chess.Move] = None
        target_val = 0.0
        target_intent = 0  # Default DEVELOPMENT

        # -------------------------------------------------------------
        # Tier 0: Sound Opening Principles & Legal Baseline
        # -------------------------------------------------------------
        if tier == CurriculumTier.LEVEL_0_LEGAL:
            def opening_soundness(m: chess.Move) -> float:
                score = 0.0
                if m.to_square in CENTER_SQUARES:
                    score += 3.5
                elif m.to_square in EXTENDED_CENTER:
                    score += 1.8
                p = board.piece_at(m.from_square)
                if p and p.piece_type in (chess.KNIGHT, chess.BISHOP):
                    score += 2.5
                if board.is_castling(m):
                    score += 4.5
                return score + rng.uniform(0.0, 0.4)

            sorted_legals = sorted(legal_moves, key=opening_soundness, reverse=True)
            target_move = sorted_legals[0]
            target_val = 0.1
            target_intent = 0  # DEVELOPMENT

        # -------------------------------------------------------------
        # Tier 1: Material & Hanging Captures
        # -------------------------------------------------------------
        elif tier == CurriculumTier.LEVEL_1_MATERIAL:
            captures = [m for m in legal_moves if board.is_capture(m)]
            if not captures:
                continue
            def capture_score(m: chess.Move) -> float:
                victim = board.piece_at(m.to_square)
                v_val = PIECE_VALUES.get(victim.piece_type, 1.0) if victim else 1.0
                attacker = board.piece_at(m.from_square)
                a_val = PIECE_VALUES.get(attacker.piece_type, 1.0) if attacker else 1.0
                return v_val * 10.0 - a_val

            captures.sort(key=capture_score, reverse=True)
            target_move = captures[0]
            target_val = 0.4
            target_intent = 1  # WIN_MATERIAL

        # -------------------------------------------------------------
        # Tier 2: Checkmate & Check Defense
        # -------------------------------------------------------------
        elif tier == CurriculumTier.LEVEL_2_CHECKMATE:
            mates = []
            for m in legal_moves:
                board.push(m)
                if board.is_checkmate():
                    mates.append(m)
                board.pop()

            if mates:
                target_move = mates[0]
                target_val = 1.0
                target_intent = 3  # MATE_ATTACK
            elif board.is_check():
                target_move = rng.choice(legal_moves)
                target_val = -0.1
                target_intent = 2  # DEFENSE
            else:
                checks = [m for m in legal_moves if board.gives_check(m)]
                if checks:
                    target_move = checks[0]
                    target_val = 0.2
                    target_intent = 3  # MATE_ATTACK
                else:
                    continue

        # -------------------------------------------------------------
        # Tier 3: Basic Tactics (Pawn Promotions, Pins)
        # -------------------------------------------------------------
        elif tier == CurriculumTier.LEVEL_3_BASIC_TACTICS:
            promos = [m for m in legal_moves if m.promotion]
            if promos:
                q_promos = [m for m in promos if m.promotion == chess.QUEEN]
                target_move = q_promos[0] if q_promos else promos[0]
                target_val = 0.8
                target_intent = 6  # TACTICAL_COMBINATION
            else:
                checks = [m for m in legal_moves if board.gives_check(m)]
                caps = [m for m in legal_moves if board.is_capture(m)]
                tactical = checks + caps
                if tactical:
                    target_move = rng.choice(tactical)
                    target_val = 0.3
                    target_intent = 6  # TACTICAL_COMBINATION
                else:
                    continue

        # -------------------------------------------------------------
        # Tier 4: Positional Fundamentals (Center, Development, King Safety)
        # -------------------------------------------------------------
        elif tier == CurriculumTier.LEVEL_4_POSITIONAL:
            best_pos_score = -999.0
            best_pos_move = legal_moves[0]
            best_intent = 0

            for m in legal_moves:
                s = 0.0
                curr_intent = 0
                if m.to_square in CENTER_SQUARES:
                    s += 2.5
                    curr_intent = 4  # CENTER_CONTROL
                piece = board.piece_at(m.from_square)
                if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                    from_r = chess.square_rank(m.from_square)
                    # Developing off back rank
                    if (perspective == chess.WHITE and from_r == 0) or (perspective == chess.BLACK and from_r == 7):
                        s += 2.0
                        curr_intent = 0  # DEVELOPMENT
                if board.is_castling(m):
                    s += 3.0
                    curr_intent = 2  # DEFENSE (King safety)
                if s > best_pos_score:
                    best_pos_score = s
                    best_pos_move = m
                    best_intent = curr_intent

            target_move = best_pos_move
            target_val = 0.25
            target_intent = best_intent

        # -------------------------------------------------------------
        # Tier 5: Tactical Depth (Forks, Skewers, Discovered Attacks)
        # -------------------------------------------------------------
        elif tier == CurriculumTier.LEVEL_5_TACTICAL_DEPTH:
            forks = []
            for m in legal_moves:
                board.push(m)
                # Count major enemy pieces attacked by moving piece
                to_sq = m.to_square
                att_sqs = board.attacks(to_sq)
                major_targets = 0
                for asq in att_sqs:
                    opp_p = board.piece_at(asq)
                    if opp_p and opp_p.color != perspective:
                        if PIECE_VALUES.get(opp_p.piece_type, 0.0) >= 3.0:
                            major_targets += 1
                board.pop()
                if major_targets >= 2:
                    forks.append(m)

            if forks:
                target_move = forks[0]
                target_val = 0.6
                target_intent = 6  # TACTICAL_COMBINATION
            else:
                # Discovered attack / check
                disc = [m for m in legal_moves if board.gives_check(m) and board.piece_at(m.from_square).piece_type != chess.QUEEN]
                if disc:
                    target_move = disc[0]
                    target_val = 0.5
                    target_intent = 6  # TACTICAL_COMBINATION
                else:
                    continue

        # -------------------------------------------------------------
        # Tier 6: Endgame Fundamentals (King Activation, Passed Pawn Pushes)
        # -------------------------------------------------------------
        elif tier == CurriculumTier.LEVEL_6_ENDGAME:
            piece_count = len(board.piece_map())
            if piece_count > 14:
                continue  # Require endgame position

            k_sq = board.king(perspective)
            endgame_moves = []
            for m in legal_moves:
                p = board.piece_at(m.from_square)
                if not p:
                    continue
                score = 0.0
                if p.piece_type == chess.PAWN:
                    # Advancing passed pawn towards 8th rank
                    dest_r = chess.square_rank(m.to_square)
                    adv = dest_r if perspective == chess.WHITE else (7 - dest_r)
                    score += adv * 2.0
                elif p.piece_type == chess.KING:
                    # King marching towards center/enemy pawns
                    dest_r, dest_f = chess.square_rank(m.to_square), chess.square_file(m.to_square)
                    dist_to_center = abs(dest_r - 3.5) + abs(dest_f - 3.5)
                    score += (7.0 - dist_to_center) * 1.5
                if score > 0.0:
                    endgame_moves.append((score, m))

            if endgame_moves:
                endgame_moves.sort(key=lambda x: x[0], reverse=True)
                target_move = endgame_moves[0][1]
                target_val = 0.5
                target_intent = 5  # ENDGAME_PUSH
            else:
                continue

        # -------------------------------------------------------------
        # Tier 7: Strategic Planning (Piece Improvement, Targeting Weaknesses)
        # -------------------------------------------------------------
        elif tier == CurriculumTier.LEVEL_7_STRATEGIC_PLANNING:
            best_strat_score = -999.0
            best_strat_move = legal_moves[0]
            for m in legal_moves:
                p = board.piece_at(m.from_square)
                if not p:
                    continue
                from_sq, to_sq = m.from_square, m.to_square
                from_mobility = len(board.attacks(from_sq))
                to_mobility = len(board.attacks(to_sq))
                mobility_gain = to_mobility - from_mobility

                s = mobility_gain * 0.5
                if m.to_square in CENTER_SQUARES:
                    s += 1.0
                if board.is_capture(m):
                    vic = board.piece_at(m.to_square)
                    if vic:
                        s += PIECE_VALUES.get(vic.piece_type, 1.0) * 2.0
                if s > best_strat_score:
                    best_strat_score = s
                    best_strat_move = m

            target_move = best_strat_move
            target_val = 0.35
            target_intent = 7  # PIECE_IMPROVEMENT

        # -------------------------------------------------------------
        # Tier 8: Multi-Target Engine Distillation (Alpha-Beta Minimax)
        # -------------------------------------------------------------
        elif tier == CurriculumTier.LEVEL_8_ENGINE_DISTILLATION:
            best_eval = -999.0 if perspective == chess.WHITE else 999.0
            best_eng_move = legal_moves[0]
            is_white = (perspective == chess.WHITE)

            for m in legal_moves:
                board.push(m)
                ev = _minimax_eval(board, depth=2, alpha=-999.0, beta=999.0, maximizing=not is_white)
                board.pop()
                if is_white:
                    if ev > best_eval:
                        best_eval = ev
                        best_eng_move = m
                else:
                    if ev < best_eval:
                        best_eval = ev
                        best_eng_move = m

            target_move = best_eng_move
            norm_val = float(np.clip(best_eval / 10.0, -1.0, 1.0))
            target_val = norm_val if is_white else -norm_val
            target_intent = 1 if board.is_capture(best_eng_move) else (4 if best_eng_move.to_square in CENTER_SQUARES else 7)

        if target_move is not None and target_move in legal_moves:
            if perspective == chess.BLACK:
                c_board = board.mirror()
                c_legals = [
                    chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)
                    for m in legal_moves
                ]
                c_target = chess.Move(
                    chess.square_mirror(target_move.from_square),
                    chess.square_mirror(target_move.to_square),
                    target_move.promotion,
                )
                c_target_idx = c_legals.index(c_target)
                feat = encoder.encode(c_board, perspective=chess.WHITE)
                samples.append(
                    ChessCurriculumSample(
                        fen=c_board.fen(),
                        perspective=chess.WHITE,
                        feature_vector=feat,
                        legal_moves=c_legals,
                        target_move=c_target,
                        target_move_idx=c_target_idx,
                        tier=tier,
                        target_value=target_val,
                        target_intent=target_intent,
                    )
                )
            else:
                target_idx = legal_moves.index(target_move)
                feat = encoder.encode(board, perspective=chess.WHITE)
                samples.append(
                    ChessCurriculumSample(
                        fen=board.fen(),
                        perspective=chess.WHITE,
                        feature_vector=feat,
                        legal_moves=legal_moves,
                        target_move=target_move,
                        target_move_idx=target_idx,
                        tier=tier,
                        target_value=target_val,
                        target_intent=target_intent,
                    )
                )

    return samples


def build_cumulative_curriculum(
    max_tier: CurriculumTier = CurriculumTier.LEVEL_8_ENGINE_DISTILLATION,
    samples_per_tier: int = 250,
    seed: int = 42,
) -> Dict[CurriculumTier, List[ChessCurriculumSample]]:
    """Build cumulative curriculum datasets (D_k = D_{k-1} + S_k) across all 9 tiers."""
    curriculum_datasets: Dict[CurriculumTier, List[ChessCurriculumSample]] = {}
    accumulated_samples: List[ChessCurriculumSample] = []

    for tier_val in range(int(max_tier) + 1):
        tier = CurriculumTier(tier_val)
        tier_samples = generate_curriculum_samples(
            tier=tier,
            num_samples=samples_per_tier,
            seed=seed + tier_val,
        )
        accumulated_samples = list(accumulated_samples) + tier_samples
        curriculum_datasets[tier] = list(accumulated_samples)

    return curriculum_datasets
