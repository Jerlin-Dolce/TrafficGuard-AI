from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st
from traffic_engine import process_video

st.set_page_config(page_title="TrafficGuard AI", page_icon="🚦", layout="wide")
st.title("🚦 TrafficGuard AI")
st.caption("Vehicle detection, tracking and configurable line-crossing review")

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    line_position = st.slider("Virtual line position (% of video height)", 10, 90, 65, 5)
    st.info("Line crossings are geometric events for human review, not proof of a traffic offence.")

uploaded = st.file_uploader("Upload a traffic video", type=["mp4","avi","mov","mkv"])
st.write("Tip: test with a short clip (20–60 seconds) first.")
if uploaded and st.button("Process video", type="primary"):
    Path("outputs").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".mp4") as tmp:
        tmp.write(uploaded.getbuffer())
        input_path = Path(tmp.name)
    try:
        with st.spinner("Loading YOLO and processing video. The first run may download model weights..."):
            result = process_video(input_path, confidence=confidence, line_position_percent=line_position)
        st.session_state["result"] = result
        st.success("Processing completed!")
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.info("Check internet access for the first model download and confirm the file is a valid video.")
    finally:
        input_path.unlink(missing_ok=True)

result = st.session_state.get("result")
if result:
    st.subheader("Analysis summary")
    a,b,c,d = st.columns(4)
    a.metric("Frames processed", f"{result['frames']:,}")
    b.metric("Unique track IDs", f"{result['unique_tracks']:,}")
    c.metric("Vehicle detections", f"{result['detections']:,}")
    d.metric("Line crossings to review", f"{result['line_crossing_events']:,}")

    vp = Path(result["video"])
    if vp.exists():
        st.subheader("Annotated video")
        st.video(str(vp))
        st.download_button("Download annotated video", vp.read_bytes(),
                           file_name="tracked_traffic.mp4", mime="video/mp4")

    dp = Path(result["detections_csv"])
    if dp.exists():
        st.subheader("Vehicle detection records")
        df = pd.read_csv(dp)
        st.dataframe(df.head(500), use_container_width=True)
        st.download_button("Download detections CSV", dp.read_bytes(),
                           file_name="vehicle_detections.csv", mime="text/csv")

    ep = Path(result["events_csv"])
    if ep.exists():
        st.subheader("Virtual line-crossing events")
        ef = pd.read_csv(ep)
        if ef.empty:
            st.info("No line-crossing events were logged for this clip.")
        else:
            st.dataframe(ef, use_container_width=True)
        st.download_button("Download events CSV", ep.read_bytes(),
                           file_name="line_crossing_events.csv", mime="text/csv")
    st.caption(f"Evidence snapshots are saved locally in: {result['evidence_dir']}")

st.divider()
st.caption("TrafficGuard AI prototype. Use authorized footage and review events before drawing conclusions.")
