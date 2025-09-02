import os
import time
import threading
from datetime import datetime
from typing import List, Tuple

import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# CONFIG
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
MODEL_PREFERENCES = ["ajeetx/spam_email_detection", "mrm8488/bert-tiny-finetuned-sms-spam-detection"]
POLL_INTERVAL_SECONDS = 30  # near-real-time polling interval

st.set_page_config(page_title="Gmail Realtime Spam Cleaner", layout="wide")
st.title("📧 Gmail Realtime Spam Cleaner (Streamlit)")

# ------------------ Utilities: Gmail API ------------------
class GmailAPIClient:
    def __init__(self):
        self.service = None
        self.creds = None

    def authenticate(self):
        if os.path.exists("token.json"):
            self.creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    st.error(f"Failed to refresh token: {e}")
                    self.creds = None
            if not self.creds:
                if not os.path.exists("credentials.json"):
                    raise FileNotFoundError("credentials.json not found. Create OAuth desktop credentials and place file next to the app.")
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                self.creds = flow.run_local_server(port=0)
            with open("token.json", "w") as f:
                f.write(self.creds.to_json())
        self.service = build("gmail", "v1", credentials=self.creds, cache_discovery=False)

    def list_message_ids(self, query: str, max_results: int = 200) -> List[str]:
        ids = []
        try:
            page_token = None
            while True:
                resp = self.service.users().messages().list(userId="me", q=query, pageToken=page_token, maxResults=500).execute()
                for m in resp.get("messages", []):
                    ids.append(m["id"])
                    if len(ids) >= max_results:
                        return ids
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            return ids
        except HttpError as e:
            st.error(f"Gmail API list error: {e}")
            return ids

    def get_subject_and_snippet(self, msg_id: str) -> Tuple[str, str]:
        try:
            msg = self.service.users().messages().get(userId="me", id=msg_id, format="metadata", metadataHeaders=["Subject","From"]).execute()
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            subject = headers.get("Subject", "(no subject)")
            snippet = msg.get("snippet", "")
            return subject, snippet
        except HttpError as e:
            return "(error)", ""

    def trash_messages(self, ids: List[str]) -> int:
        if not ids:
            return 0
        count = 0
        CHUNK = 100
        try:
            for i in range(0, len(ids), CHUNK):
                batch = self.service.new_batch_http_request()
                for mid in ids[i:i+CHUNK]:
                    batch.add(self.service.users().messages().trash(userId="me", id=mid))
                batch.execute()
                count += len(ids[i:i+CHUNK])
            return count
        except HttpError as e:
            st.error(f"Error moving messages to Trash: {e}")
            return count

# ------------------ Spam model loader ------------------
@st.cache_resource(show_spinner=False)
def load_spam_pipeline():
    last_exc = None
    # Honor optional Hugging Face token for private models
    hf_token = st.session_state.get("hf_token")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    load_kwargs = {}
    if hf_token:
        # Support both old/new transformers/huggingface-hub arguments
        load_kwargs["use_auth_token"] = hf_token
        load_kwargs["token"] = hf_token
    for model_id in MODEL_PREFERENCES:
        try:
            pipe = pipeline("text-classification", model=model_id, truncation=True, **load_kwargs)
            st.session_state["model_id"] = model_id
            return pipe
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Failed to load any spam model. Last error: {last_exc}")

# ------------------ App state ------------------
if "gmail_client" not in st.session_state:
    st.session_state.gmail_client = GmailAPIClient()

if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "scanning" not in st.session_state:
    st.session_state.scanning = False

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    st.write("Near-real-time: polling every N seconds (uses Gmail API).")
    poll = st.number_input("Poll interval (seconds)", min_value=10, max_value=600, value=POLL_INTERVAL_SECONDS, step=5)
    st.session_state.poll_interval = int(poll)
    max_msgs = st.number_input("Max messages per scan", min_value=50, max_value=2000, value=300, step=50)
    st.session_state.max_messages = int(max_msgs)
    threshold = st.slider("Spam confidence threshold", min_value=0.5, max_value=0.99, value=0.9, step=0.01)
    st.session_state.threshold = float(threshold)
    auto_trash = st.checkbox("Auto-trash predicted spam (when scanned)", value=False)
    st.session_state.auto_trash = auto_trash
    st.markdown("---")
    st.write("Model preference (tried in order):")
    st.write(", ".join(MODEL_PREFERENCES))
    st.markdown("---")
    # Optional Hugging Face token for private model access
    hf_token_input = st.text_input("Hugging Face token (optional)", type="password")
    if hf_token_input:
        st.session_state.hf_token = hf_token_input.strip()
        st.caption("Hugging Face token saved for this session.")
    st.markdown("---")
    # Credential helper: allow uploading credentials.json directly in the app
    cred_file = st.file_uploader("Upload Google OAuth credentials.json", type=["json"], accept_multiple_files=False)
    if cred_file is not None:
        try:
            content = cred_file.getvalue()
            with open("credentials.json", "wb") as f:
                f.write(content)
            st.success("Saved credentials.json next to the app.")
        except Exception as e:
            st.error(f"Failed to save credentials.json: {e}")
    # Also allow restoring an existing token.json to avoid re-consent
    token_upload = st.file_uploader("(Optional) Upload token.json", type=["json"], accept_multiple_files=False, key="token_uploader")
    if token_upload is not None:
        try:
            content = token_upload.getvalue()
            with open("token.json", "wb") as f:
                f.write(content)
            st.success("Saved token.json.")
        except Exception as e:
            st.error(f"Failed to save token.json: {e}")
    st.markdown("---")
    if st.button("Authenticate with Google (Gmail API)"):
        try:
            st.session_state.gmail_client.authenticate()
            st.success("Authenticated with Gmail API. Scanning will work.")
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Auth failed: {e}")

# Load model (cached)
with st.spinner("Loading spam model..."):
    try:
        pipe = load_spam_pipeline()
        st.success(f"Loaded model: {st.session_state.get('model_id','unknown')}")
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

# Controls
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Scan controls")
    if st.button("Scan Now"):
        st.session_state.run_scan_now = True
    run_live = st.checkbox("Enable live polling (near-real-time)", value=True)
    st.session_state.live_polling = run_live

with col2:
    st.subheader("Status")
    last = st.session_state.last_scan_time
    st.write(f"Last scan: {last if last else 'Never'}")
    st.write(f"Messages scanned in last result: {len(st.session_state.last_results)}")

# ------------------ Scanning logic ------------------
def classify_batch(pipe, texts):
    # returns list of (label, score)
    try:
        out = pipe(texts, truncation=True)
        res = []
        for o in out:
            label = o.get("label", "")
            score = float(o.get("score", 0.0))
            res.append((label, score))
        return res
    except Exception as e:
        st.error(f"Classification error: {e}")
        return [("ERROR", 0.0)] * len(texts)

def run_scan():
    if not st.session_state.gmail_client.service:
        st.warning("Not authenticated. Please authenticate with Google (sidebar).")
        return
    st.session_state.scanning = True
    try:
        # Query both read and unread messages in inbox and anywhere: adjust as needed
        q = "in:inbox"
        ids = st.session_state.gmail_client.list_message_ids(q, max_results=st.session_state.max_messages)
        if not ids:
            st.info("No messages found for the query.")
            st.session_state.last_results = []
            st.session_state.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.scanning = False
            return

        texts = []
        id_map = []
        for mid in ids:
            subj, snippet = st.session_state.gmail_client.get_subject_and_snippet(mid)
            text = (subj or "") + " " + (snippet or "")
            texts.append(text)
            id_map.append((mid, subj))

        # classify in batches
        BATCH = 32
        results = []
        spam_ids = []
        for i in range(0, len(texts), BATCH):
            batch_texts = texts[i:i+BATCH]
            batch_map = id_map[i:i+BATCH]
            classified = classify_batch(pipe, batch_texts)
            for (label, score), (mid, subj) in zip(classified, batch_map):
                lab = label.lower() if isinstance(label, str) else str(label)
                results.append({"id": mid, "subject": subj, "label": label, "score": score})
                if ("spam" in lab) or ("label_1" in lab) or (lab.strip() in ["spam","spam_label"]):
                    if score >= st.session_state.threshold:
                        spam_ids.append(mid)

        st.session_state.last_results = results
        st.session_state.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # auto-trash if enabled
        if st.session_state.auto_trash and spam_ids:
            n = st.session_state.gmail_client.trash_messages(spam_ids)
            st.success(f"Auto-moved {n} predicted spam messages to Trash.")
    except Exception as e:
        st.error(f"Scan failed: {e}")
    finally:
        st.session_state.scanning = False

# Trigger scan when requested or via polling
if st.session_state.get("run_scan_now", False):
    run_scan()
    st.session_state.run_scan_now = False

# Live polling using streamlit-autorefresh (safe)
from streamlit_autorefresh import st_autorefresh
if st.session_state.live_polling:
    refresh_count = st_autorefresh(interval=st.session_state.poll_interval * 1000, key="poller")
    if refresh_count:
        run_scan()

# ------------------ Display results ------------------
st.markdown("## Scan Results (most recent)")
results = st.session_state.get("last_results", [])
if results:
    import pandas as pd
    df = pd.DataFrame(results)
    df_display = df.sort_values("score", ascending=False).head(500)
    st.dataframe(df_display.reset_index(drop=True))
    # Buttons to delete selected high-confidence spam
    to_delete = st.multiselect("Select message IDs to move to Trash (choose IDs from table)", options=df_display["id"].tolist())
    if st.button("Move selected to Trash"):
        if to_delete:
            n = st.session_state.gmail_client.trash_messages(to_delete)
            st.success(f"Moved {n} messages to Trash.")
        else:
            st.info("No message IDs selected.")
else:
    st.info("No recent scan results. Click 'Scan Now' or authenticate to begin.")

st.markdown("---")
st.markdown("**Important notes:**")
st.markdown("- This app uses Gmail API (oauth desktop client) and `gmail.modify` scope to move messages to Trash.")
st.markdown("- 'Near-real-time' here is implemented via polling every N seconds (set in the sidebar). For true push notifications you'd need Gmail Pub/Sub + a server endpoint (more complex).")
st.markdown("- Review predictions before deleting; models can give false positives.")
