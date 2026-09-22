"""Command line interface: streamcount run | serve | find-stream | download-model | report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .detector import MODEL_URLS, ensure_model
from .pipeline import RunConfig, run
from .sources import find_stream_in_page


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("source (pick one)")
    group.add_argument("--url", help="HLS (.m3u8), RTSP, or direct video URL")
    group.add_argument("--youtube", metavar="URL", help="YouTube live or recording link")
    group.add_argument("--video", type=Path, help="local video file")
    group.add_argument("--image", type=Path, help="single image")
    group.add_argument("--images", type=Path, metavar="DIR", help="folder of images (frame sequence)")
    group.add_argument("--headers", default="",
                       help='extra ffmpeg headers, e.g. "Referer=https://example.com/;User-Agent=Mozilla/5.0"')
    group.add_argument("--seek", type=float, default=0.0, help="start offset in seconds (recordings)")
    group.add_argument("--yt-js-runtime", metavar="RUNTIME[:PATH]", default=None,
                       help="JS runtime for yt-dlp (e.g. node:C:/path/node.exe) when YouTube requires it")


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("engines")
    group.add_argument("--engine", choices=["yolo", "vlm", "both"], default="yolo",
                       help="yolo = local ONNX detector (free, offline); vlm = vision LLM (API key); "
                            "both = same frame through both, for comparison")
    group.add_argument("--target", default="people",
                       help='what to count: "people", "cars", or "people,cars" (default: people)')
    group.add_argument("--model", type=Path, default=None,
                       help="custom ONNX model (default: auto-download yolov8n-pose)")
    group.add_argument("--conf", type=float, default=0.25)
    group.add_argument("--iou", type=float, default=0.5)
    group.add_argument("--tiles", type=int, default=1,
                       help="split the frame into NxN tiles (recovers small/distant objects; N=1 off)")
    group.add_argument("--tile-overlap", type=float, default=0.15)
    group.add_argument("--vlm-key", default=None, help="API key (else STREAMCOUNT_VLM_KEY/OPENROUTER_API_KEY/.env)")
    group.add_argument("--vlm-base-url", default="https://openrouter.ai/api/v1")
    group.add_argument("--vlm-model", default="google/gemini-2.5-flash-lite")


def _add_flow_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("flow mode — count who PASSES (each person once)")
    group.add_argument("--flow", action="store_true",
                       help="enable unique-pass counting: tracking with stable ids")
    group.add_argument("--flow-min-hits", type=int, default=3,
                       help="detections required to confirm a track (3 = conservative)")
    group.add_argument("--flow-min-move", type=float, default=40.0,
                       help="displacement in px required to count as a passer-by (standing people never count)")
    group.add_argument("--flow-max-age", type=float, default=5.0,
                       help="seconds without a detection before a track dies (ghost re-id lasts 15 s)")
    group.add_argument("--vlm-check", type=float, default=0.0,
                       help="run the VLM every N seconds as a recall anchor (0 = off)")


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument("--interval", type=float, default=2.0, help="seconds between sampled frames")
    group.add_argument("--frames", type=int, default=0,
                       help="number of frames to process (0 = 10 for streams, all for images)")
    group.add_argument("--timeline", choices=["auto", "wall", "video"], default="auto",
                       help="time base for events: wall clock (live streams) or video time (recordings)")
    group.add_argument("--out", type=Path, default=Path("runs"), help="output directory")
    group.add_argument("--annotate", action="store_true", help="save annotated frames")
    group.add_argument("--keep-frames", type=int, default=0, metavar="N",
                       help="keep only the last N annotated frames on disk (0 = keep all; "
                            "with --annotate; the dashboard uses 10)")
    group.add_argument("--timelapse", action="store_true",
                       help="write an annotated timelapse MP4 of the run (into the run folder)")
    group.add_argument("--timelapse-fps", type=float, default=12.0,
                       help="timelapse playback fps (default 12 = 24x at --interval 2)")
    group.add_argument("--tag", default="", help="suffix for the run folder name")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streamcount",
        description="Count people and vehicles in any live stream or recording "
                    "(YouTube, HLS, RTSP, files) with a local detector or a vision LLM.",
    )
    parser.add_argument("--version", action="version", version=f"streamcount {__version__}")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="run a counting job")
    _add_source_args(run_parser)
    _add_engine_args(run_parser)
    _add_flow_args(run_parser)
    _add_output_args(run_parser)

    find_parser = sub.add_parser("find-stream",
                                 help="extract a fresh HLS URL from a webcam page (tokens expire)")
    find_parser.add_argument("page", help="page URL that embeds the player")
    find_parser.add_argument("--cam-id", default=None, help="camera id to filter (e.g. 4280)")

    model_parser = sub.add_parser("download-model", help="pre-download ONNX model weights")
    model_parser.add_argument("--variant", default="yolov8n-pose", choices=sorted(MODEL_URLS))

    report_parser = sub.add_parser(
        "report", help="summarise a finished run: per-minute rates, peak, optional chart PNG"
    )
    report_parser.add_argument("run_dir", type=Path, help="run folder produced by `run`")
    report_parser.add_argument("--chart", type=Path, default=None,
                               help="write a per-minute bar/cumulative chart PNG here")

    serve_parser = sub.add_parser(
        "serve", help="local live dashboard: paste a link, watch the count running")
    serve_parser.add_argument("--port", type=int, default=8766, help="localhost port (default 8766)")
    serve_parser.add_argument("--run-root", type=Path, default=Path("runs"),
                              help="where dashboard runs are written (default: runs)")
    serve_parser.add_argument("--open", action="store_true", help="open the page in your browser")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "find-stream":
        print(find_stream_in_page(args.page, cam_id=args.cam_id))
        return 0

    if args.command == "download-model":
        path = ensure_model(args.variant)
        print(f"model ready: {path}")
        return 0

    if args.command == "report":
        from .report import build_report

        report = build_report(args.run_dir, chart_path=args.chart)
        if not report.get("events_file"):
            print(f"no events file found in {args.run_dir} (is it a --flow run?)")
            return 2
        duration = f" in {report['duration_min']} min" if report["duration_min"] else ""
        print(f"passers-by: {report['passes']}{duration}")
        print(f"rate: mean {report['mean_per_minute']}/min | peak {report['peak_per_minute']}/min")
        if report.get("recall"):
            print(f"detector recall ~{report['recall'] * 100:.0f}% -> recall-corrected estimate "
                  f"~{report['passes_recall_corrected']}")
        print("per-minute: " + ", ".join(str(v) for v in report["passes_per_minute"]))
        if report.get("chart"):
            print(f"chart: {report['chart']}")
        return 0

    if args.command == "serve":
        from .webapp import serve as serve_dashboard

        serve_dashboard(port=args.port, runs_root=args.run_root, open_browser=args.open)
        return 0

    if args.command != "run":
        parser.print_help()
        return 2

    if not any([args.url, args.youtube, args.video, args.image, args.images]):
        parser.error("provide a source: --url, --youtube, --video, --image or --images")

    config = RunConfig(
        url=args.youtube or args.url,
        video=args.video,
        image=args.image,
        images_dir=args.images,
        yt_dlp_js_runtime=args.yt_js_runtime,
        headers=args.headers,
        seek_s=args.seek,
        interval=args.interval,
        frames=args.frames,
        timeline=args.timeline,
        engine=args.engine,
        target=args.target,
        model=args.model,
        conf=args.conf,
        iou=args.iou,
        tiles=args.tiles,
        tile_overlap=args.tile_overlap,
        vlm_key=args.vlm_key,
        vlm_base_url=args.vlm_base_url,
        vlm_model=args.vlm_model,
        flow=args.flow,
        flow_min_hits=args.flow_min_hits,
        flow_min_move=args.flow_min_move,
        flow_max_age=args.flow_max_age,
        vlm_check=args.vlm_check,
        out_dir=args.out,
        annotate=args.annotate,
        keep_frames=args.keep_frames,
        timelapse=args.timelapse,
        timelapse_fps=args.timelapse_fps,
        tag=args.tag,
    )
    result = run(config)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
