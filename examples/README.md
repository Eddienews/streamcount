# Examples

| script | what it shows |
|---|---|
| `earthcam_hour.sh` | fresh token extraction + 1-hour flow counting with a VLM recall anchor on a public webcam |
| `youtube_live.sh` | YouTube live stream → vision-LLM per-frame counting |
| `replay_parameters.sh` | the deterministic method: capture frames once, replay parameter sets on identical input |

All scripts assume `streamcount` is installed (`pip install -e .`) and — for the VLM modes —
an API key in `OPENROUTER_API_KEY` or `STREAMCOUNT_VLM_KEY`.
