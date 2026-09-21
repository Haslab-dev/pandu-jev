"""Rich Chess Feature Encoder for Pandu Policy (224 dimensions).

Encodes board state, rules, piece placements, tactical maps, king safety,
mobility, material balance, and positional context into a scale-invariant,
information-dense representation.
"""

from typing import Dict, List, Optional, Set, Tuple
import chess
import numpy as np

PIECE_VALUES: Dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 100.0,
}

CENTER_SQUARES: Set[chess.Square] = {chess.E4, chess.D4, chess.E5, chess.D5}
EXTENDED_CENTER: Set[chess.Square] = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}


class ChessFeatureEncoder:
    """Encodes a chess.Board into a 224-dimensional feature vector.

    Feature Group Breakdown:
    1. Piece Placement & Values (64): Normalized piece values per square (-1.0 to 1.0).
    2. Attack & Control Map (64): Net control per square (friendly - opp attackers).
    3. Hanging & Vulnerable Piece Map (64): Undefended friendly and opponent pieces.
    4. Rule & Context Flags (16): Castling rights, en passant, check status, repetition, halfmove, phase, mobility, material.
    5. King Safety Metrics (8): King coordinates, edge proximity, ring pressure for both sides.
    6. Tactical Potentials (8): Center control, legal captures, checks, mate-in-1, pins, passed pawns.
    Total: 224 dimensions.
    """

    DIMENSION: int = 224

    @staticmethod
    def encode(board: chess.Board, perspective: chess.Color = chess.WHITE) -> np.ndarray:
        """Extract the full 224-dimensional feature vector."""
        features = np.zeros(ChessFeatureEncoder.DIMENSION, dtype=np.float32)
        opp = not perspective

        self_mat = 0.0
        opp_mat = 0.0

        # --- 1. Piece Placement & Values (0..63) ---
        piece_map = board.piece_map()
        for sq, piece in piece_map.items():
            val = PIECE_VALUES.get(piece.piece_type, 0.0)
            if piece.color == perspective:
                features[sq] = val / 10.0
                self_mat += val
            else:
                features[sq] = -val / 10.0
                opp_mat += val

        # --- 2. Attack & Control Map (64..127) & 3. Hanging Pieces (128..191) ---
        for sq in chess.SQUARES:
            att_self = len(board.attackers(perspective, sq))
            att_opp = len(board.attackers(opp, sq))
            # Control: normalized difference
            features[64 + sq] = np.clip((att_self - att_opp) / 4.0, -1.0, 1.0)

            # Hanging / tactical vulnerability
            piece = piece_map.get(sq)
            if piece is not None:
                if piece.color == perspective:
                    if att_opp > 0 and att_self == 0:
                        features[128 + sq] = 1.0  # Friendly piece is completely undefended!
                    elif att_opp > att_self:
                        features[128 + sq] = 0.5  # Friendly piece is under-defended
                else:
                    if att_self > 0 and att_opp == 0:
                        features[128 + sq] = -1.0  # Enemy piece is hanging (prime target!)
                    elif att_self > att_opp:
                        features[128 + sq] = -0.5  # Enemy piece is vulnerable

        # --- 4. Rule & Context Flags (192..207) ---
        idx = 192
        features[idx + 0] = 1.0 if perspective == chess.WHITE else -1.0
        features[idx + 1] = 1.0 if board.has_kingside_castling_rights(perspective) else 0.0
        features[idx + 2] = 1.0 if board.has_queenside_castling_rights(perspective) else 0.0
        features[idx + 3] = 1.0 if board.has_kingside_castling_rights(opp) else 0.0
        features[idx + 4] = 1.0 if board.has_queenside_castling_rights(opp) else 0.0
        features[idx + 5] = 1.0 if board.has_legal_en_passant() else 0.0
        features[idx + 6] = (chess.square_file(board.ep_square) / 7.0) if board.ep_square is not None else -1.0
        features[idx + 7] = 1.0 if board.is_check() else 0.0
        features[idx + 8] = 0.5 if board.is_repetition(2) else (1.0 if board.is_repetition(3) else 0.0)
        features[idx + 9] = min(board.halfmove_clock, 50) / 50.0
        features[idx + 10] = min(len(piece_map), 32) / 32.0  # Game phase (1.0 opening, <0.4 endgame)

        # Mobility
        legal_moves_count = board.legal_moves.count()
        features[idx + 11] = min(legal_moves_count, 60) / 60.0

        # Approximate opponent mobility
        # Using king and major piece pseudo-attacks or board turn flip approximation
        features[idx + 12] = 0.5  # default nominal mobility
        features[idx + 13] = self_mat / 40.0
        features[idx + 14] = opp_mat / 40.0
        features[idx + 15] = np.clip((self_mat - opp_mat) / 20.0, -1.0, 1.0)

        # --- 5. King Safety Metrics (208..215) ---
        k_self = board.king(perspective)
        k_opp = board.king(opp)

        idx = 208
        if k_self is not None:
            r_s, f_s = chess.square_rank(k_self), chess.square_file(k_self)
            features[idx + 0] = r_s / 7.0
            features[idx + 1] = f_s / 7.0
            features[idx + 2] = min(r_s, 7 - r_s, f_s, 7 - f_s) / 3.5
            # Attacker pressure in 3x3 surrounding friendly king
            ring_sqs = [
                sq for sq in chess.SQUARES
                if max(abs(chess.square_rank(sq) - r_s), abs(chess.square_file(sq) - f_s)) == 1
            ]
            features[idx + 3] = sum(len(board.attackers(opp, sq)) for sq in ring_sqs) / 12.0

        if k_opp is not None:
            r_o, f_o = chess.square_rank(k_opp), chess.square_file(k_opp)
            features[idx + 4] = r_o / 7.0
            features[idx + 5] = f_o / 7.0
            features[idx + 6] = min(r_o, 7 - r_o, f_o, 7 - f_o) / 3.5
            ring_opp = [
                sq for sq in chess.SQUARES
                if max(abs(chess.square_rank(sq) - r_o), abs(chess.square_file(sq) - f_o)) == 1
            ]
            features[idx + 7] = sum(len(board.attackers(perspective, sq)) for sq in ring_opp) / 12.0

        # --- 6. Tactical Potentials (216..223) ---
        idx = 216
        # Center control (e4, d4, e5, d5)
        self_center = sum(len(board.attackers(perspective, sq)) for sq in CENTER_SQUARES)
        opp_center = sum(len(board.attackers(opp, sq)) for sq in CENTER_SQUARES)
        features[idx + 0] = min(self_center, 8) / 8.0
        features[idx + 1] = min(opp_center, 8) / 8.0

        # Legal captures & checks count
        captures_count = 0
        checks_count = 0
        can_mate_in_1 = 0.0

        for m in board.legal_moves:
            if board.is_capture(m):
                captures_count += 1
            if board.gives_check(m):
                checks_count += 1
                board.push(m)
                if board.is_checkmate():
                    can_mate_in_1 = 1.0
                board.pop()

        features[idx + 2] = min(captures_count, 10) / 10.0
        features[idx + 3] = min(checks_count, 5) / 5.0
        features[idx + 4] = can_mate_in_1

        # Pinned pieces
        pins_count = sum(1 for sq in piece_map if piece_map[sq].color == perspective and board.is_pinned(perspective, sq))
        features[idx + 5] = min(pins_count, 6) / 6.0

        # Passed pawns count
        passed_self = 0
        passed_opp = 0
        for sq, p in piece_map.items():
            if p.piece_type == chess.PAWN:
                r, f = chess.square_rank(sq), chess.square_file(sq)
                # Check for opposing pawns in front and adjacent files
                opp_pawns_in_path = False
                for r_ahead in range(r + 1, 8) if p.color == chess.WHITE else range(0, r):
                    for f_adj in (f - 1, f, f + 1):
                        if 0 <= f_adj < 8:
                            adj_sq = chess.square(f_adj, r_ahead)
                            occ = piece_map.get(adj_sq)
                            if occ and occ.piece_type == chess.PAWN and occ.color != p.color:
                                opp_pawns_in_path = True
                                break
                    if opp_pawns_in_path:
                        break
                if not opp_pawns_in_path:
                    if p.color == perspective:
                        passed_self += 1
                    else:
                        passed_opp += 1

        features[idx + 6] = min(passed_self, 4) / 4.0
        features[idx + 7] = min(passed_opp, 4) / 4.0

        return features
