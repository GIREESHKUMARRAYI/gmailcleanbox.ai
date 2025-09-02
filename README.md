Gmail Realtime Spam Cleaner (Streamlit)
======================================

This Streamlit app connects to Gmail via the Gmail API (OAuth Desktop flow),
polls your mailbox near-real-time (by default every 30 seconds), classifies
messages using a pretrained spam model from Hugging Face, and optionally moves
predicted spam to Trash.

Files:
- app.py            : Streamlit application (main file)
- requirements.txt  : Python dependencies
- run.sh            : helper launcher (Unix)

Setup:
1. Create OAuth credentials on Google Cloud Console:
   - Application type: Desktop
   - Download the JSON and name it 'credentials.json'
   - Place credentials.json in the same folder as app.py
   - Alternatively, run the app and upload the file via the sidebar uploader

2. Install dependencies (recommended in virtualenv):
   pip install -r requirements.txt

3. Run the app:
   streamlit run app.py

4. In the app sidebar, click 'Authenticate with Google (Gmail API)'. A browser window will open
   to let you sign in and grant the gmail.modify scope.
   - You can also upload an existing token.json to skip re-consent on the same account.

5. Enable live polling and/or click 'Scan Now'. Review results and choose to move predicted spam to Trash.

Notes:
- The app tries models in this order: ajeetx/spam_email_detection, then mrm8488/bert-tiny-finetuned-sms-spam-detection.
- Polling is used to approximate real-time behavior. For push notifications, use Gmail Pub/Sub + a server + an HTTPS endpoint.
- Trash is recoverable until permanently deleted.
 - If you need to access private Hugging Face models, paste your token in the sidebar field before model loading.
