"""Flow tracking: count each person exactly once ("who passed"), not per frame.

SORT-lite design, tuned for low-frame-rate sampling (1 frame every 1-5 s):

* association by IoU OR centre distance (time gaps between samples are big,
  so IoU alone is too strict for walking people),
* a track is *confirmed* only after ``min_hits`` detections,
* a confirmed track is *counted* once it has moved ``min_move_px`` from where it
  was first seen -- people who just stand still (vendors, buskers) never count
  as "passers-by",
* dead tracks become *ghosts* for ``ghost_s`` seconds; a new detection that
  reappears nearby inherits the ghost's id, so a few frames of detector drop-out
  do not double-count the same person.
"""

from __future__ import annotations

from collections.abc import Iterable

Box = tuple[float, float, float, float]


class FlowTracker:
    def __init__(
        self,
        iou_thr: float = 0.15,
        max_age_s: float = 5.0,
        min_hits: int = 3,
        min_move_px: float = 40.0,
        ghost_s: float = 15.0,
        ghost_dist_px: float = 140.0,
    ) -> None:
        self.iou_thr = iou_thr
        self.max_age_s = max_age_s
        self.min_hits = min_hits
        self.min_move_px = min_move_px
        self.ghost_s = ghost_s
        self.ghost_dist_px = ghost_dist_px

        self.tracks: dict[int, dict] = {}
        self.ghosts: list[dict] = []
        self.next_id = 1

        self.passes: list[dict] = []   # one event per person counted
        self.n_short = 0               # tracks that died before being confirmed (noise)
        self.n_static = 0              # confirmed tracks that never moved (standing people)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _centre(box: Box) -> tuple[float, float]:
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    # ---------------------------------------------------------------- counting
    def _maybe_count(self, t: float, tr: dict) -> dict | None:
        if tr["counted"] or tr["hits"] < self.min_hits or tr["disp"] < self.min_move_px:
            return None
        tr["counted"] = True
        event = {
            "t": round(t, 1),
            "id": tr["id"],
            "hits": tr["hits"],
            "disp": round(tr["disp"]),
            "n": len(self.passes) + 1,
        }
        self.passes.append(event)
        return event

    # ------------------------------------------------------------------- cycle
    def update(self, t: float, dets: Iterable[Box]) -> tuple[list[dict], list[dict]]:
        """Feed one sampled frame's detections.

        Returns ``(live_tracks, new_events)`` where ``new_events`` are the people
        counted on this frame.
        """
        dets = list(dets)
        events: list[dict] = []

        # 1) greedy association: detection <-> track by IoU or centre distance
        pairs: list[tuple[float, int, int]] = []
        for tid, tr in self.tracks.items():
            for di, det in enumerate(dets):
                iou = self._iou(tr["box"], det)
                (cx1, cy1), (cx2, cy2) = self._centre(tr["box"]), self._centre(det)
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                radius = 0.9 * max(tr["box"][3] - tr["box"][1], det[3] - det[1])
                if iou >= self.iou_thr or dist <= radius:
                    pairs.append((iou + 0.3 * (1 - dist / (radius + 1e-6)), tid, di))
        pairs.sort(reverse=True)

        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            tr, det = self.tracks[tid], dets[di]
            (cx, cy) = self._centre(det)
            tr["box"] = det
            tr["hits"] += 1
            tr["last_t"] = t
            tr["disp"] = max(
                tr["disp"],
                ((cx - tr["first_c"][0]) ** 2 + (cy - tr["first_c"][1]) ** 2) ** 0.5,
            )
            used_t.add(tid)
            used_d.add(di)
            event = self._maybe_count(t, tr)
            if event:
                events.append(event)

        # 2) unmatched tracks age out -> ghosts (re-identification pool)
        for tid in list(self.tracks):
            if tid not in used_t and t - self.tracks[tid]["last_t"] > self.max_age_s:
                tr = self.tracks.pop(tid)
                self.ghosts.append(
                    {"box": tr["box"], "t": t, "id": tr["id"], "counted": tr["counted"]}
                )
                if not tr["counted"]:
                    if tr["hits"] >= self.min_hits:
                        self.n_static += 1
                    else:
                        self.n_short += 1
        self.ghosts = [g for g in self.ghosts if t - g["t"] <= self.ghost_s]

        # 3) unmatched detections: inherit a recent ghost's id, or be born new
        for di, det in enumerate(dets):
            if di in used_d:
                continue
            best, best_d = None, 1e18
            for g in self.ghosts:
                (cx1, cy1), (cx2, cy2) = self._centre(g["box"]), self._centre(det)
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                if dist <= self.ghost_dist_px and dist < best_d:
                    best, best_d = g, dist
            if best is not None:
                tid, counted = best["id"], best["counted"]
                self.ghosts.remove(best)
            else:
                tid, counted = self.next_id, False
                self.next_id += 1
            self.tracks[tid] = {
                "id": tid,
                "box": det,
                "last_t": t,
                "hits": 1,
                "first_c": self._centre(det),
                "disp": 0.0,
                "counted": counted,
            }

        return list(self.tracks.values()), events

    def finalize(self, t: float) -> None:
        """Account for tracks still alive when the run ends (reporting only)."""
        for tr in self.tracks.values():
            if tr["counted"]:
                continue
            if tr["hits"] >= self.min_hits and tr["disp"] < self.min_move_px:
                self.n_static += 1
            else:
                self.n_short += 1

    # ------------------------------------------------------------------ summary
    @property
    def total(self) -> int:
        return len(self.passes)
