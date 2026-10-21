# Palm Payment

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the application:

```powershell
$env:PALMPAY_SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
$env:PALMPAY_ADMIN_EMAIL = "admin@example.com"
$env:PALMPAY_ADMIN_PASSWORD = "replace-with-a-strong-unique-password"
python app.py
```

The app generates a temporary session key if `PALMPAY_SECRET_KEY` is not set;
set it as above to keep sessions valid across restarts. The built-in first-install
admin login is disabled unless both `PALMPAY_ADMIN_EMAIL` and
`PALMPAY_ADMIN_PASSWORD` are configured. Alternatively, create an administrator
through the admin registration page.

The SQLite database is created locally as `palm_payment.db`. It is intentionally
excluded from Git so local account and transaction data are not shared.

## Palm recognition model

The CCNet ONNX model is required for palm registration and verification. It is
excluded from Git because it is about 245 MB. Keep it at
`models/palm_ccnet.onnx`, or set `PALM_CCNET_MODEL_PATH` to its location before
starting the app:

```powershell
$env:PALM_CCNET_MODEL_PATH = "C:\path\to\palm_ccnet.onnx"
python app.py
```
