from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import uuid
import json
import math
import hashlib
import re
import secrets
import time
import base64
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)

app.secret_key = os.getenv("PALMPAY_SECRET_KEY") or secrets.token_hex(32)


# =========================================================
# SEPARATE CUSTOMER / MERCHANT PORTAL MODES
# =========================================================

AUTH_SAFE_ENDPOINTS = {
    "home",
    "login",
    "register",
    "merchant_login",
    "merchant_register",
    "logout",
    "verify_receipt",
    "receipt_api",
    "static",
}


@app.before_request
def enforce_portal_mode():
    mode = session.get("portal_mode")
    if not mode:
        return None

    endpoint = request.endpoint or ""

    if endpoint in AUTH_SAFE_ENDPOINTS:
        return None

    if mode == "merchant":
        if (
            endpoint == "merchant_dashboard"
            or request.path.startswith("/api/merchant")
            or endpoint == "security_live_monitor"
        ):
            return None
        return redirect(url_for("merchant_dashboard"))

    if mode == "customer":
        if endpoint == "merchant_dashboard" or request.path.startswith("/api/merchant"):
            return redirect(url_for("dashboard"))
        return None

    if mode == "admin":
        if endpoint == "admin_dashboard" or request.path.startswith("/api/admin"):
            return None
        return redirect(url_for("admin_dashboard"))

    return None


# =========================================================
# CCNET CONTACTLESS PALM RECOGNITION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CCNET_MODEL_PATH = Path(
    os.getenv("PALM_CCNET_MODEL_PATH", str(PROJECT_ROOT / "models" / "palm_ccnet.onnx"))
)
# Demo threshold only. Calibrate this against your own webcam captures.
CCNET_SIMILARITY_THRESHOLD = float(os.getenv("PALM_CCNET_THRESHOLD", "0.60"))
_ccnet_session = None


def get_ccnet_session():
    """Load the CCNet ONNX model lazily so the Flask app can still start."""
    global _ccnet_session

    if _ccnet_session is not None:
        return _ccnet_session

    if ort is None:
        raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime")

    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for CCNet inference.")

    if not CCNET_MODEL_PATH.exists():
        raise RuntimeError(f"CCNet model not found: {CCNET_MODEL_PATH}")

    _ccnet_session = ort.InferenceSession(
        str(CCNET_MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )
    return _ccnet_session


def _decode_image_payload(image_payload):
    """Decode a browser data URL/base64 image into an OpenCV image."""
    if not isinstance(image_payload, str) or not image_payload.strip():
        raise ValueError("Palm image is missing.")

    raw = image_payload.strip()
    if raw.startswith("data:"):
        if "," not in raw:
            raise ValueError("Invalid palm image data URL.")
        raw = raw.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 palm image.") from exc

    if not raw_bytes:
        raise ValueError("Palm image is empty.")

    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode palm image.")

    return image


def ccnet_embedding_from_image(image_payload):
    """Convert one palm crop into the 2048-D CCNet embedding."""
    image = _decode_image_payload(image_payload)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    tensor = gray.astype(np.float32) / 255.0
    tensor = tensor[None, None, :, :]  # [1, 1, 128, 128]

    session_ = get_ccnet_session()
    input_name = session_.get_inputs()[0].name
    output = session_.run(None, {input_name: tensor})[0]
    embedding = np.asarray(output, dtype=np.float32).reshape(-1)

    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        raise ValueError("CCNet returned an invalid zero embedding.")

    return (embedding / norm).tolist()


def average_ccnet_embeddings(embeddings):
    if not embeddings:
        return None

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 2048:
        raise ValueError("Invalid CCNet embedding dimensions.")

    average = matrix.mean(axis=0)
    norm = float(np.linalg.norm(average))
    if norm <= 1e-12:
        raise ValueError("Unable to create a stable CCNet registration template.")

    return (average / norm).tolist()


def cosine_similarity(left, right):
    left_np = np.asarray(left, dtype=np.float32)
    right_np = np.asarray(right, dtype=np.float32)

    if left_np.size != 2048 or right_np.size != 2048:
        raise ValueError("Invalid CCNet embedding length.")

    left_norm = np.linalg.norm(left_np)
    right_norm = np.linalg.norm(right_np)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0

    return float(np.dot(left_np, right_np) / (left_norm * right_norm))


# =========================================================
# DATABASE
# =========================================================

DB_PATH = PROJECT_ROOT / "palm_payment.db"


def get_db():
    # Always use the database inside the project folder. This prevents Flask
    # from accidentally creating/using a different palm_payment.db when the
    # app is started from another working directory.
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mobile TEXT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            palm_id TEXT,
            palm_template TEXT,
            account_status TEXT NOT NULL DEFAULT 'ACTIVE'
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            receiver TEXT NOT NULL,
            receiver_name TEXT,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            transaction_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            receiver TEXT NOT NULL,
            amount REAL NOT NULL,
            risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            reasons TEXT,
            action TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            amount REAL,
            transaction_id TEXT,
            is_read INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS security_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS virtual_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            card_number TEXT NOT NULL UNIQUE,
            expiry TEXT NOT NULL,
            cvv TEXT NOT NULL,
            is_frozen INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            merchant_id TEXT NOT NULL UNIQUE,
            business_name TEXT NOT NULL,
            category TEXT NOT NULL,
            accepting_payments INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Advanced merchant features: product catalog, POS sales and refunds.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            sku TEXT,
            category TEXT NOT NULL DEFAULT 'Other',
            price REAL NOT NULL DEFAULT 0,
            stock INTEGER NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(merchant_user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_pos_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_user_id INTEGER NOT NULL,
            order_id TEXT NOT NULL UNIQUE,
            customer_mobile TEXT,
            customer_name TEXT,
            amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'COMPLETED',
            transaction_id TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(merchant_user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_pos_sale_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pos_sale_id INTEGER NOT NULL,
            product_id INTEGER,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            line_total REAL NOT NULL,
            FOREIGN KEY(pos_sale_id) REFERENCES merchant_pos_sales(id),
            FOREIGN KEY(product_id) REFERENCES merchant_products(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_refunds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_user_id INTEGER NOT NULL,
            transaction_id TEXT NOT NULL,
            refund_id TEXT NOT NULL UNIQUE,
            original_amount REAL NOT NULL,
            refund_amount REAL NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'COMPLETED',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(merchant_user_id) REFERENCES users(id)
        )
    """)



    # =====================================================
    # ADVANCED SECURITY / FAMILY PALMS / ADMIN TABLES
    # =====================================================
    conn.execute("""
        CREATE TABLE IF NOT EXISTS palm_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            profile_name TEXT NOT NULL,
            relationship TEXT NOT NULL DEFAULT 'Family',
            palm_id TEXT NOT NULL UNIQUE,
            palm_template TEXT NOT NULL,
            permissions TEXT NOT NULL DEFAULT '{"payment":true}',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS palm_security_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 0,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Administrator',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_ref TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migrate existing databases without deleting any current records.
    # CREATE TABLE IF NOT EXISTS does not add columns to an existing table.
    def add_column_if_missing(table_name, column_name, column_definition):
        existing_columns = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            )

    add_column_if_missing("users", "security_version", "INTEGER NOT NULL DEFAULT 1")
    add_column_if_missing("users", "payment_pin", "TEXT")
    add_column_if_missing("users", "palm_template", "TEXT")
    add_column_if_missing("users", "mobile", "TEXT")
    add_column_if_missing("users", "account_status", "TEXT NOT NULL DEFAULT 'ACTIVE'")

    # Merchant schema migration. The user's existing database was created
    # before accepting_payments was added, so add it when it is missing.
    add_column_if_missing(
        "merchant_profiles",
        "accepting_payments",
        "INTEGER NOT NULL DEFAULT 1"
    )
    add_column_if_missing(
        "merchant_profiles",
        "created_at",
        "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    )
    add_column_if_missing(
        "merchant_profiles",
        "updated_at",
        "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    )

    # Existing transaction tables may also need this field.
    add_column_if_missing("transactions", "receiver_name", "TEXT")

    conn.commit()
    conn.close()


# Initialize / upgrade the database as soon as this module is imported.
# This is important when Flask is started with `flask run`, where
# `if __name__ == "__main__"` is not executed.
init_db()


# =========================================================
# INDIAN MOBILE NUMBER HELPERS
# =========================================================

def normalize_indian_mobile(value):

    if value is None:
        return None

    digits = re.sub(r"\D", "", str(value).strip())

    # Accept 9876543210, 919876543210 and +91 9876543210.
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]

    if re.fullmatch(r"[6-9]\d{9}", digits):
        return digits

    return None


def format_indian_mobile(value):

    mobile = normalize_indian_mobile(value)

    if not mobile:
        return value or ""

    return "+91 " + mobile


@app.before_request
def enforce_session_security():
    if "user_id" not in session:
        return None

    public_endpoints = {"home", "login", "register", "register_palm", "save_palm", "static"}
    if request.endpoint in public_endpoints:
        return None

    conn = get_db()
    user = conn.execute("SELECT security_version FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()

    if not user:
        session.clear()
        return redirect(url_for("login"))

    db_version = int(user["security_version"] or 1)
    session_version = session.get("security_version")

    # Upgrade older sessions once without forcing existing users to log in again.
    if session_version is None:
        session["security_version"] = db_version
        return None

    if int(session_version) != db_version:
        session.clear()
        return redirect(url_for("login"))

    return None


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        mobile = normalize_indian_mobile(request.form.get("mobile", ""))
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not mobile or not email or not password:

            return render_template(
                "register.html",
                error="Please enter your name, valid Indian mobile number, email and password."
            )

        if not name or not email or not password:

            return render_template(
                "register.html",
                error="Please fill all the fields."
            )

        if len(password) < 6:

            return render_template(
                "register.html",
                error="Password must contain at least 6 characters."
            )

        hashed_password = generate_password_hash(password)

        conn = get_db()

        try:

            cursor = conn.execute("""
                INSERT INTO users
                (name, mobile, email, password)
                VALUES (?, ?, ?, ?)
            """, (
                name,
                mobile,
                email,
                hashed_password
            ))

            conn.commit()

            user_id = cursor.lastrowid

            conn.close()

            session.clear()
            session["portal_mode"] = "customer"
            session["user_id"] = user_id
            session["security_version"] = 1
            session["user_name"] = name

            return redirect(url_for("register_palm"))

        except sqlite3.IntegrityError:

            conn.close()

            return render_template(
                "register.html",
                error="Email already registered!"
            )

    return render_template("register.html")


# =========================================================
# PALM REGISTRATION PAGE
# =========================================================

@app.route("/register-palm")
def register_palm():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "register_palm.html",
        name=session.get("user_name")
    )


# =========================================================
# SAVE PALM TEMPLATE
# =========================================================

@app.route("/save-palm", methods=["POST"])
def save_palm():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "Please login first."
        }), 401

    data = request.get_json(silent=True) or {}

    legacy_template = data.get("template")
    handedness = data.get("handedness", "Unknown")
    palm_images = data.get("palm_images") or []

    if legacy_template is not None:
        if not isinstance(legacy_template, list) or len(legacy_template) != 63:
            return jsonify({
                "success": False,
                "message": "Invalid landmark template."
            }), 400

    if not isinstance(palm_images, list):
        return jsonify({
            "success": False,
            "message": "Invalid palm image samples."
        }), 400

    palm_images = [item for item in palm_images if isinstance(item, str) and item.strip()]
    if len(palm_images) > 10:
        palm_images = palm_images[:10]

    ccnet_embeddings = []

    if palm_images:
        try:
            for image_payload in palm_images:
                ccnet_embeddings.append(ccnet_embedding_from_image(image_payload))
            ccnet_template = average_ccnet_embeddings(ccnet_embeddings)
        except Exception as exc:
            app.logger.exception("CCNet registration failed")
            return jsonify({
                "success": False,
                "message": f"CCNet palm registration failed: {exc}"
            }), 500
    else:
        ccnet_template = None

    if ccnet_template is None and legacy_template is None:
        return jsonify({
            "success": False,
            "message": "No palm biometric samples received."
        }), 400

    palm_id = "PALM-" + uuid.uuid4().hex[:10].upper()

    stored_payload = {
        "version": 2,
        "method": "CCNet_2048D" if ccnet_template is not None else "MediaPipe_63D_LEGACY",
        "template": legacy_template,
        "ccnet_embedding": ccnet_template,
        "ccnet_samples": len(ccnet_embeddings),
        "handedness": handedness
    }

    conn = get_db()

    conn.execute("""
        UPDATE users
        SET palm_id = ?,
            palm_template = ?
        WHERE id = ?
    """, (
        palm_id,
        json.dumps(stored_payload),
        session["user_id"]
    ))

    conn.execute("UPDATE palm_profiles SET active = 0, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND relationship = 'Primary'", (session["user_id"],))
    conn.execute("""
        INSERT INTO palm_profiles
        (user_id, profile_name, relationship, palm_id, palm_template, permissions, active)
        VALUES (?, ?, 'Primary', ?, ?, ?, 1)
    """, (
        session["user_id"],
        "Primary Palm",
        palm_id,
        json.dumps(stored_payload),
        json.dumps({"payment": True})
    ))

    conn.commit()
    conn.close()

    session["palm_id"] = palm_id

    log_security_activity(
        session["user_id"],
        "Palm registered",
        f"Palm biometric template created using {'CCNet' if ccnet_template is not None else 'legacy landmark'} mode"
    )

    return jsonify({
        "success": True,
        "message": "Palm registered successfully.",
        "palm_id": palm_id,
        "biometric_mode": "CCNet_2048D" if ccnet_template is not None else "MediaPipe_63D_LEGACY",
        "samples": len(ccnet_embeddings) if ccnet_embeddings else 0
    })


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        conn = get_db()

        user = conn.execute("""
            SELECT *
            FROM users
            WHERE email = ?
        """, (
            email,
        )).fetchone()

        conn.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            session["portal_mode"] = "customer"
            session["user_id"] = user["id"]
            session["security_version"] = int(user["security_version"] or 1)
            session["user_name"] = user["name"]
            session["user_mobile"] = user["mobile"]
            session["palm_id"] = user["palm_id"]

            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid email or password!"
        )

    return render_template("login.html")


# =========================================================
# MERCHANT REGISTRATION
# =========================================================

@app.route("/merchant-register", methods=["GET", "POST"])
def merchant_register():

    if request.method == "POST":
        owner_name = request.form.get("owner_name", "").strip()
        business_name = request.form.get("business_name", "").strip()
        category = request.form.get("category", "Other").strip() or "Other"
        mobile = normalize_indian_mobile(request.form.get("mobile", ""))
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        allowed_categories = {"Retail", "Food & Beverage", "Education", "Services", "Freelance", "Other"}
        if category not in allowed_categories:
            category = "Other"

        if not owner_name or not business_name or not mobile or not email or not password:
            return render_template(
                "merchant_register.html",
                error="Please complete all merchant registration fields."
            )

        if len(business_name) > 60:
            return render_template("merchant_register.html", error="Business name must be 60 characters or less.")

        if len(password) < 6:
            return render_template("merchant_register.html", error="Password must contain at least 6 characters.")

        conn = get_db()
        try:
            # The schema is created/upgraded by init_db(), including mobile and
            # security_version columns required by the merchant flow.
            cursor = conn.execute("""
                INSERT INTO users (name, mobile, email, password)
                VALUES (?, ?, ?, ?)
            """, (
                owner_name,
                mobile,
                email,
                generate_password_hash(password)
            ))

            user_id = cursor.lastrowid
            merchant_id = "MCH-" + uuid.uuid4().hex[:10].upper()
            now = time.strftime("%Y-%m-%d %H:%M:%S")

            conn.execute("""
                INSERT INTO merchant_profiles
                (user_id, merchant_id, business_name, category, accepting_payments, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            """, (
                user_id,
                merchant_id,
                business_name,
                category,
                now,
                now
            ))

            conn.commit()

        except sqlite3.IntegrityError:
            conn.rollback()
            return render_template(
                "merchant_register.html",
                error="That email is already registered. Use a different email for the merchant account."
            )

        except sqlite3.Error as exc:
            conn.rollback()
            app.logger.exception("Merchant registration database error")
            return render_template(
                "merchant_register.html",
                error=f"Merchant registration database error: {exc}. Restart the Flask app after replacing app.py."
            ), 500

        finally:
            conn.close()

        session.clear()
        session["portal_mode"] = "merchant"
        session["user_id"] = user_id
        session["security_version"] = 1
        session["user_name"] = owner_name
        session["merchant_id"] = merchant_id
        return redirect(url_for("merchant_dashboard"))

    return render_template("merchant_register.html")


# =========================================================
# MERCHANT LOGIN
# =========================================================

@app.route("/merchant-login", methods=["GET", "POST"])
def merchant_login():

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute("""
            SELECT u.*, mp.merchant_id, mp.business_name, mp.category, mp.accepting_payments
            FROM users u
            JOIN merchant_profiles mp ON mp.user_id = u.id
            WHERE u.email = ?
            LIMIT 1
        """, (email,)).fetchone()
        conn.close()

        if user and user["account_status"] == "SUSPENDED":
            return render_template("merchant_login.html", error="This merchant account has been suspended by the PalmPay security team.")

        if user and check_password_hash(user["password"], password):
            session.clear()
            session["portal_mode"] = "merchant"
            session["user_id"] = user["id"]
            session["security_version"] = int(user["security_version"] or 1)
            session["user_name"] = user["name"]
            session["user_mobile"] = user["mobile"]
            session["merchant_id"] = user["merchant_id"]
            return redirect(url_for("merchant_dashboard"))

        return render_template("merchant_login.html", error="Invalid merchant email or password, or no merchant account exists for this email.")

    return render_template("merchant_login.html")


# =========================================================
# MERCHANT DASHBOARD
# =========================================================

@app.route("/merchant/dashboard")
def merchant_dashboard():

    if session.get("portal_mode") != "merchant" or "user_id" not in session:
        return redirect(url_for("merchant_login"))

    conn = get_db()
    profile = conn.execute("""
        SELECT merchant_id, business_name, category, accepting_payments, created_at, updated_at
        FROM merchant_profiles
        WHERE user_id = ?
        LIMIT 1
    """, (session["user_id"],)).fetchone()
    conn.close()

    if not profile:
        return redirect(url_for("merchant_register"))

    return render_template(
        "merchant_dashboard.html",
        name=session.get("user_name", "Merchant"),
        profile=serialize_merchant_profile(profile)
    )


# =========================================================
# ADVANCED PALM LIVENESS + CONTINUOUS AUTHENTICATION
# =========================================================

@app.route("/api/palm/challenge", methods=["POST"])
def palm_challenge():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Please login first."}), 401

    token = secrets.token_urlsafe(32)
    now = time.time()
    # Prototype challenge: user must visibly move the palm left/right while
    # keeping an open palm. The server validates the submitted liveness
    # measurements and challenge freshness.
    session["palm_challenge_token"] = token
    session["palm_challenge_issued_at"] = now
    session["palm_challenge_expires_at"] = now + 45
    session["palm_challenge_action"] = "MOVE_LEFT_RIGHT"

    return jsonify({
        "success": True,
        "challenge_token": token,
        "action": "MOVE_LEFT_RIGHT",
        "instruction": "Move your open palm slowly left and right inside the frame.",
        "expires_in": 45
    })


@app.route("/api/continuous-auth/heartbeat", methods=["GET"])
def continuous_auth_heartbeat():
    if "user_id" not in session:
        return jsonify({"success": False, "authenticated": False, "message": "Login required."}), 401

    now = time.time()
    last_biometric = session.get("continuous_palm_verified_at")
    max_age = 300  # 5 minutes for this project prototype
    age = None if not last_biometric else max(0, now - float(last_biometric))
    reauth_required = last_biometric is None or age > max_age

    return jsonify({
        "success": True,
        "authenticated": not reauth_required,
        "reauth_required": reauth_required,
        "age_seconds": round(age, 1) if age is not None else None,
        "reauth_after_seconds": max_age,
        "message": "Palm re-verification required." if reauth_required else "Continuous authentication active."
    })


@app.route("/api/security/live-monitor", methods=["GET"])
def security_live_monitor():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401

    conn = get_db()
    risk_rows = conn.execute("""
        SELECT risk_score, risk_level, reasons, action, receiver, amount, created_at
        FROM risk_events
        WHERE user_id = ?
          AND datetime(created_at) >= datetime('now', '-15 minutes')
        ORDER BY id DESC LIMIT 12
    """, (session["user_id"],)).fetchall()

    activity_rows = conn.execute("""
        SELECT activity, details, created_at
        FROM security_activity
        WHERE user_id = ?
          AND datetime(created_at) >= datetime('now', '-15 minutes')
        ORDER BY id DESC LIMIT 12
    """, (session["user_id"],)).fetchall()
    conn.close()

    events = []
    for row in risk_rows:
        try:
            reasons = json.loads(row["reasons"] or "[]")
        except Exception:
            reasons = []
        events.append({
            "type": "risk",
            "level": row["risk_level"],
            "score": row["risk_score"],
            "action": row["action"],
            "amount": row["amount"],
            "receiver": row["receiver"],
            "reasons": reasons,
            "created_at": row["created_at"]
        })
    for row in activity_rows:
        text = (row["activity"] or "").lower()
        if any(word in text for word in ("failed", "incorrect", "blocked")):
            events.append({
                "type": "security",
                "level": "HIGH",
                "score": None,
                "action": row["activity"],
                "amount": None,
                "receiver": None,
                "reasons": [row["details"] or row["activity"]],
                "created_at": row["created_at"]
            })

    events.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    high_count = sum(1 for e in events if e.get("level") == "HIGH")
    medium_count = sum(1 for e in events if e.get("level") == "MEDIUM")
    status = "HIGH ALERT" if high_count else ("MONITORING" if medium_count else "SECURE")

    return jsonify({
        "success": True,
        "status": status,
        "high_alerts": high_count,
        "medium_alerts": medium_count,
        "events": events[:15]
    })





def validate_palm_security_challenge(challenge_token, liveness_metrics, require_fresh=True):
    """Validate the demo anti-spoof movement challenge and return metrics."""
    expected_token = session.get("palm_challenge_token")
    expires_at = session.get("palm_challenge_expires_at")
    action = session.get("palm_challenge_action")

    try:
        movement_score = float((liveness_metrics or {}).get("movement_score", 0))
        direction_changes = int((liveness_metrics or {}).get("direction_changes", 0))
        movement_samples = int((liveness_metrics or {}).get("movement_samples", 0))
        image_count = int((liveness_metrics or {}).get("image_count", 0))
        live_frames = int((liveness_metrics or {}).get("live_frames", 0))
    except (TypeError, ValueError):
        movement_score, direction_changes, movement_samples = 0.0, 0, 0
        image_count, live_frames = 0, 0

    token_ok = bool(challenge_token and expected_token and secrets.compare_digest(str(challenge_token), str(expected_token)))
    freshness_ok = True
    if require_fresh:
        freshness_ok = bool(action == "MOVE_LEFT_RIGHT" and expires_at and time.time() <= float(expires_at))

    # Presentation-attack heuristic score: higher means more suspicious.
    motion_quality = min(1.0, movement_score / 0.055)
    direction_quality = min(1.0, direction_changes / 2.0)
    sample_quality = min(1.0, movement_samples / 10.0)
    image_quality = min(1.0, image_count / 6.0)
    frame_quality = min(1.0, live_frames / 4.0)
    spoof_score = round(100 * (
        0.34 * (1 - motion_quality) +
        0.23 * (1 - direction_quality) +
        0.18 * (1 - sample_quality) +
        0.15 * (1 - image_quality) +
        0.10 * (1 - frame_quality)
    ), 1)

    challenge_ok = (
        token_ok and freshness_ok and
        movement_score >= 0.015 and
        direction_changes >= 1 and
        movement_samples >= 4 and
        image_count >= 4 and
        live_frames >= 2
    )

    return {
        "valid": bool(challenge_ok),
        "spoof_score": spoof_score,
        "movement_score": round(movement_score, 4),
        "direction_changes": direction_changes,
        "movement_samples": movement_samples,
        "image_count": image_count,
        "live_frames": live_frames,
    }


def log_palm_security_event(user_id, event_type, score, details=""):
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO palm_security_events (user_id, event_type, score, details) VALUES (?, ?, ?, ?)",
            (user_id, event_type, float(score), details)
        )
        conn.commit()
        conn.close()
    except Exception:
        app.logger.exception("Unable to log palm security event")


def record_admin_audit(action, target="", details=""):
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO admin_audit (admin_ref, action, target, details) VALUES (?, ?, ?, ?)",
            (session.get("admin_email", "admin"), action, target, details)
        )
        conn.commit()
        conn.close()
    except Exception:
        app.logger.exception("Unable to record admin audit")


# =========================================================
# PALM VERIFICATION
# =========================================================


@app.route("/verify-palm", methods=["POST"])
def verify_palm():
    if "user_id" not in session:
        return jsonify({"success": False, "authenticated": False, "message": "Please login first."}), 401

    data = request.get_json(silent=True) or {}
    live_template = data.get("template")
    palm_images = data.get("palm_images") or []
    is_live = bool(data.get("is_live", False))
    gesture_ok = bool(data.get("gesture_ok", False))
    challenge_token = str(data.get("challenge_token", "")).strip()
    liveness_metrics = data.get("liveness_metrics") or {}

    if live_template is not None and (not isinstance(live_template, list) or len(live_template) != 63):
        live_template = None
    if not isinstance(palm_images, list):
        palm_images = []
    palm_images = [x for x in palm_images if isinstance(x, str) and x.strip()][:10]
    if not live_template and not palm_images:
        return jsonify({"success": False, "authenticated": False, "message": "No palm biometric data received."}), 400

    security = validate_palm_security_challenge(challenge_token, liveness_metrics)
    if not security["valid"]:
        log_security_activity(session["user_id"], "Palm anti-spoof check failed", f"Presentation risk score {security['spoof_score']} · movement={security['movement_score']} · direction changes={security['direction_changes']}")
        log_palm_security_event(session["user_id"], "ANTI_SPOOF_FAIL", security["spoof_score"], "Movement challenge did not provide enough live variation.")
        return jsonify({
            "success": False,
            "authenticated": False,
            "message": "Anti-spoof check failed. Move your open palm slowly left and right and try again.",
            "anti_spoof": security
        }), 403

    if not is_live or not gesture_ok:
        log_palm_security_event(session["user_id"], "LIVENESS_FAIL", security["spoof_score"], "Open-palm or liveness state was not satisfied.")
        return jsonify({
            "success": False,
            "authenticated": False,
            "message": "Please show a live open palm inside the frame.",
            "anti_spoof": security
        }), 403

    conn = get_db()
    candidates = []
    user = conn.execute("SELECT palm_id, palm_template FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if user and user["palm_template"]:
        try:
            saved_data = json.loads(user["palm_template"])
            candidates.append({
                "profile_id": 0,
                "profile_name": "Primary Palm",
                "relationship": "Primary",
                "permissions": {"payment": True},
                "payload": saved_data
            })
        except Exception:
            pass

    family_rows = conn.execute("""
        SELECT id, profile_name, relationship, palm_template, permissions, active
        FROM palm_profiles
        WHERE user_id = ? AND active = 1
        ORDER BY id DESC
    """, (session["user_id"],)).fetchall()
    conn.close()

    for row in family_rows:
        try:
            payload = json.loads(row["palm_template"])
            permissions = json.loads(row["permissions"] or '{"payment":true}')
            candidates.append({
                "profile_id": int(row["id"]),
                "profile_name": row["profile_name"],
                "relationship": row["relationship"],
                "permissions": permissions,
                "payload": payload
            })
        except Exception:
            continue

    if not candidates:
        return jsonify({"success": False, "authenticated": False, "message": "No registered palm found. Please register a palm first."}), 400

    best = None
    # Prefer CCNet whenever camera samples are available. The registration and
    # verification cameras use the same palm crop path for this prototype.
    if palm_images:
        try:
            live_embeddings = [ccnet_embedding_from_image(item) for item in palm_images]
            live_ccnet = average_ccnet_embeddings(live_embeddings)
            for candidate in candidates:
                saved_ccnet = candidate["payload"].get("ccnet_embedding")
                if not saved_ccnet:
                    continue
                similarity = cosine_similarity(saved_ccnet, live_ccnet)
                if best is None or similarity > best["similarity"]:
                    best = {"candidate": candidate, "similarity": similarity, "biometric_mode": "CCNet_2048D", "sample_count": len(live_embeddings)}
        except Exception as exc:
            app.logger.exception("CCNet verification failed")
            return jsonify({"success": False, "authenticated": False, "message": f"CCNet palm verification failed: {exc}"}), 500

    # Legacy fallback for old accounts where only MediaPipe 63D template exists.
    if best is None and isinstance(live_template, list):
        for candidate in candidates:
            saved_template = candidate["payload"].get("template")
            if not isinstance(saved_template, list) or len(saved_template) != len(live_template):
                continue
            total = 0.0
            for saved, live in zip(saved_template, live_template):
                difference = saved - live
                total += difference * difference
            distance = math.sqrt(total / len(saved_template))
            score = 1.0 - min(1.0, distance / 0.12)
            if best is None or score > best["score"]:
                best = {"candidate": candidate, "distance": distance, "score": score, "biometric_mode": "MediaPipe_63D_LEGACY", "sample_count": 1}

    if best is None:
        return jsonify({"success": False, "authenticated": False, "message": "No compatible registered palm template found."}), 400

    candidate = best["candidate"]
    payment_allowed = bool(candidate["permissions"].get("payment", True))

    if best.get("biometric_mode") == "CCNet_2048D":
        similarity = float(best["similarity"])
        matched = similarity >= CCNET_SIMILARITY_THRESHOLD
    else:
        distance = float(best["distance"])
        matched = distance <= 0.12

    if not matched:
        metric = f"similarity={best.get('similarity', 0):.4f}" if best.get("similarity") is not None else f"distance={best.get('distance', 0):.4f}"
        log_security_activity(session["user_id"], "Palm verification failed", f"Best profile {candidate['profile_name']} · {metric}")
        log_palm_security_event(session["user_id"], "PALM_MATCH_FAIL", float(best.get("similarity", best.get("score", 0)) * 100), metric)
        return jsonify({
            "success": True,
            "authenticated": False,
            "message": "Palm does not match any registered palm profile.",
            "profile_name": candidate["profile_name"],
            "anti_spoof": security,
            "threshold": CCNET_SIMILARITY_THRESHOLD
        })

    if not payment_allowed:
        log_security_activity(session["user_id"], "Palm permission blocked", f"Matched {candidate['profile_name']} but payment permission is disabled")
        return jsonify({
            "success": False,
            "authenticated": False,
            "message": f"{candidate['profile_name']} is verified, but payment permission is disabled.",
            "profile_name": candidate["profile_name"],
            "anti_spoof": security
        }), 403

    verified_now = time.time()
    session["palm_verified_at"] = verified_now
    session["palm_verification_token"] = secrets.token_urlsafe(32)
    session["continuous_palm_verified_at"] = verified_now
    session["matched_palm_profile_id"] = candidate["profile_id"]
    session["matched_palm_profile_name"] = candidate["profile_name"]
    session.pop("palm_challenge_token", None)
    session.pop("palm_challenge_issued_at", None)
    session.pop("palm_challenge_expires_at", None)
    session.pop("palm_challenge_action", None)

    metric_text = f"similarity {best['similarity']:.4f}" if best.get("similarity") is not None else f"landmark distance {best['distance']:.4f}"
    log_security_activity(session["user_id"], "Palm verified", f"{candidate['profile_name']} · {metric_text} · anti-spoof risk {security['spoof_score']}")
    log_palm_security_event(session["user_id"], "PALM_MATCH_SUCCESS", float(100 - security["spoof_score"]), f"{candidate['profile_name']} · {metric_text}")

    return jsonify({
        "success": True,
        "authenticated": True,
        "profile_name": candidate["profile_name"],
        "relationship": candidate["relationship"],
        "biometric_mode": best["biometric_mode"],
        "similarity": round(best["similarity"], 4) if best.get("similarity") is not None else None,
        "distance": round(best["distance"], 4) if best.get("distance") is not None else None,
        "threshold": CCNET_SIMILARITY_THRESHOLD,
        "anti_spoof": security,
        "message": f"Palm authenticated successfully using {candidate['profile_name']}."
    })


# =========================================================
# FIND USER BY MOBILE
# =========================================================

@app.route("/api/find-user", methods=["GET"])
@app.route("/lookup-receiver", methods=["GET"])
def find_user():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "Please login first."
        }), 401

    mobile = normalize_indian_mobile(
        request.args.get("mobile", "")
    )

    if not mobile:
        return jsonify({
            "success": False,
            "message": "Enter a valid Indian mobile number."
        }), 400

    conn = get_db()

    user = conn.execute("""
        SELECT id, name, mobile
        FROM users
        WHERE mobile = ?
        LIMIT 1
    """, (mobile,)).fetchone()

    conn.close()

    if not user:
        return jsonify({
            "success": False,
            "message": "No PalmPay user found with this mobile number."
        }), 404

    if user["id"] == session["user_id"]:
        return jsonify({
            "success": False,
            "message": "You cannot send money to your own account."
        }), 400

    return jsonify({
        "success": True,
        "name": user["name"],
        "mobile": user["mobile"]
    })


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        return redirect(url_for("login"))

    conn = get_db()

    user = conn.execute("""
        SELECT *
        FROM users
        WHERE id = ?
    """, (
        session["user_id"],
    )).fetchone()

    transactions = conn.execute("""
        SELECT *
        FROM transactions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 20
    """, (
        session["user_id"],
    )).fetchall()

    notifications = conn.execute("""
        SELECT *
        FROM notifications
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 30
    """, (
        session["user_id"],
    )).fetchall()

    unread_notifications = conn.execute("""
        SELECT COUNT(*) AS total
        FROM notifications
        WHERE user_id = ?
        AND is_read = 0
    """, (
        session["user_id"],
    )).fetchone()["total"]

    spent_result = conn.execute("""
        SELECT COALESCE(SUM(amount), 0) AS total
        FROM transactions
        WHERE user_id = ?
        AND status = 'SUCCESS'
    """, (
        session["user_id"],
    )).fetchone()

    total_spent = spent_result["total"]

    topup_result = conn.execute("""
        SELECT COALESCE(SUM(ABS(amount)), 0) AS total
        FROM transactions
        WHERE user_id = ?
        AND status = 'TOPUP'
    """, (
        session["user_id"],
    )).fetchone()

    total_added = topup_result["total"]

    conn.close()

    starting_balance = 10000

    balance = (
        starting_balance
        + total_added
        - total_spent
    )

    return render_template(
        "dashboard.html",
        user=user,
        name=user["name"],
        mobile=user["mobile"],
        palm_id=user["palm_id"],
        transactions=transactions,
        balance=balance,
        total_spent=total_spent,
        total_added=total_added,
        notifications=notifications,
        unread_notifications=unread_notifications,
        payment_error=session.pop("payment_error_message", None)
    )


# =========================================================
# ADD MONEY
# =========================================================

@app.route("/add-money", methods=["POST"])
def add_money():

    if "user_id" not in session:
        return redirect(url_for("login"))

    amount_text = request.form.get(
        "amount",
        ""
    ).strip()

    try:
        amount = float(amount_text)
    except ValueError:
        return redirect(url_for("dashboard"))

    if amount <= 0:
        return redirect(url_for("dashboard"))

    conn = get_db()

    transaction_id = (
        "TOPUP-"
        + uuid.uuid4().hex[:12].upper()
    )

    conn.execute("""
        INSERT INTO transactions
        (
            user_id,
            receiver,
            receiver_name,
            amount,
            status,
            transaction_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session["user_id"],
        "Wallet Top Up",
        "Wallet Top Up",
        -amount,
        "TOPUP",
        transaction_id
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("dashboard"))


# =========================================================
# PAYMENT RISK ANALYSIS
# =========================================================

def payment_risk_analysis(conn, user_id, receiver, amount):
    """Explainable rule-based fraud/risk prototype for the college project."""
    score = 0
    reasons = []

    # Amount rules
    if amount >= 50000:
        score += 45
        reasons.append("Very high transaction amount")
    elif amount >= 25000:
        score += 30
        reasons.append("High transaction amount")
    elif amount >= 10000:
        score += 15
        reasons.append("Higher-than-normal transaction amount")

    # Recent payment frequency
    recent = conn.execute("""
        SELECT COUNT(*) AS count
        FROM transactions
        WHERE user_id = ?
          AND status = 'SUCCESS'
          AND datetime(created_at) >= datetime('now', '-10 minutes')
    """, (user_id,)).fetchone()["count"]
    if recent >= 5:
        score += 25
        reasons.append("Many payments in the last 10 minutes")
    elif recent >= 3:
        score += 12
        reasons.append("Several recent payments")

    # New receiver
    previous_to_receiver = conn.execute("""
        SELECT COUNT(*) AS count
        FROM transactions
        WHERE user_id = ? AND receiver = ? AND status = 'SUCCESS'
    """, (user_id, receiver)).fetchone()["count"]
    if previous_to_receiver == 0:
        score += 15
        reasons.append("First payment to this receiver")

    # Compare against recent average
    avg_row = conn.execute("""
        SELECT AVG(amount) AS avg_amount
        FROM transactions
        WHERE user_id = ? AND status = 'SUCCESS'
          AND datetime(created_at) >= datetime('now', '-30 days')
    """, (user_id,)).fetchone()
    avg_amount = float(avg_row["avg_amount"] or 0)
    if avg_amount > 0 and amount >= avg_amount * 5:
        score += 20
        reasons.append("Amount is much higher than your recent average")
    elif avg_amount > 0 and amount >= avg_amount * 3:
        score += 10
        reasons.append("Amount is above your recent average")

    # Failed security events in the last 30 minutes
    failed = conn.execute("""
        SELECT COUNT(*) AS count
        FROM security_activity
        WHERE user_id = ?
          AND (activity LIKE '%failed%' OR activity LIKE '%Incorrect%')
          AND datetime(created_at) >= datetime('now', '-30 minutes')
    """, (user_id,)).fetchone()["count"]
    if failed >= 3:
        score += 20
        reasons.append("Multiple recent failed security attempts")
    elif failed >= 1:
        score += 8
        reasons.append("Recent failed security attempt detected")

    score = min(100, int(score))
    if score >= 60:
        level = "HIGH"
    elif score >= 35:
        level = "MEDIUM"
    else:
        level = "LOW"

    if not reasons:
        reasons.append("No unusual activity detected")
    return score, level, reasons


def payment_risk_key(receiver, amount):
    raw = f"{receiver}|{amount:.2f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =========================================================
# PAYMENT
# =========================================================

@app.route("/pay", methods=["POST"])
def pay():

    if "user_id" not in session:
        return redirect(url_for("login"))

    # A successful palm scan authorizes one payment only.
    verified_at = session.get("palm_verified_at")

    if not verified_at or (time.time() - float(verified_at)) > 60:
        session.pop("palm_verified_at", None)
        session.pop("palm_verification_token", None)
        session.pop("payment_pin_verified_at", None)
        return redirect(url_for("dashboard"))

    # Payment PIN is the second authorization step. It is verified
    # through the server-side endpoint before this form is submitted.
    pin_verified_at = session.get("payment_pin_verified_at")
    if not pin_verified_at or (time.time() - float(pin_verified_at)) > 60:
        session.pop("payment_pin_verified_at", None)
        session.pop("payment_otp_verified_at", None)
        return redirect(url_for("dashboard"))

    otp_verified_at = session.get("payment_otp_verified_at")
    if not otp_verified_at or (time.time() - float(otp_verified_at)) > 120:
        session.pop("payment_otp_verified_at", None)
        return redirect(url_for("dashboard"))

    receiver = normalize_indian_mobile(
        request.form.get("receiver", "")
    )

    amount_text = request.form.get(
        "amount",
        ""
    ).strip()

    if not receiver:
        session.pop("palm_verified_at", None)
        session.pop("palm_verification_token", None)
        session.pop("payment_pin_verified_at", None)
        return redirect(url_for("dashboard"))

    try:
        amount = float(amount_text)
    except (ValueError, TypeError):
        session.pop("palm_verified_at", None)
        session.pop("palm_verification_token", None)
        session.pop("payment_pin_verified_at", None)
        return redirect(url_for("dashboard"))

    if not math.isfinite(amount) or amount <= 0:
        session.pop("palm_verified_at", None)
        session.pop("palm_verification_token", None)
        session.pop("payment_pin_verified_at", None)
        return redirect(url_for("dashboard"))

    conn = get_db()

    # Resolve receiver name on the server instead of trusting the browser.
    receiver_user = conn.execute("""
        SELECT id, name, mobile
        FROM users
        WHERE mobile = ?
    """, (
        receiver,
    )).fetchone()

    if not receiver_user:
        conn.close()
        session.pop("palm_verified_at", None)
        session.pop("palm_verification_token", None)
        session.pop("payment_pin_verified_at", None)
        return redirect(url_for("dashboard"))

    merchant_profile = conn.execute("""
        SELECT merchant_id, business_name, accepting_payments
        FROM merchant_profiles
        WHERE user_id = ?
        LIMIT 1
    """, (receiver_user["id"],)).fetchone()

    if merchant_profile and not merchant_profile["accepting_payments"]:
        conn.close()
        session.pop("palm_verified_at", None)
        session.pop("palm_verification_token", None)
        session.pop("payment_pin_verified_at", None)
        session.pop("payment_otp_verified_at", None)
        session["payment_error_message"] = "This merchant is currently not accepting payments."
        return redirect(url_for("dashboard"))

    # Prevent sending to the same account.
    if receiver_user["id"] == session["user_id"]:
        conn.close()
        session.pop("palm_verified_at", None)
        session.pop("palm_verification_token", None)
        session.pop("payment_pin_verified_at", None)
        return redirect(url_for("dashboard"))

    spent_result = conn.execute("""
        SELECT COALESCE(SUM(amount), 0) AS total
        FROM transactions
        WHERE user_id = ?
        AND status = 'SUCCESS'
    """, (
        session["user_id"],
    )).fetchone()

    total_spent = spent_result["total"]

    topup_result = conn.execute("""
        SELECT COALESCE(SUM(ABS(amount)), 0) AS total
        FROM transactions
        WHERE user_id = ?
        AND status = 'TOPUP'
    """, (
        session["user_id"],
    )).fetchone()

    total_added = topup_result["total"]

    balance = (
        10000
        + total_added
        - total_spent
    )

    if amount > balance:
        conn.close()
        session.pop("palm_verified_at", None)
        session.pop("palm_verification_token", None)
        session.pop("payment_pin_verified_at", None)
        return redirect(url_for("dashboard"))

    risk_score, risk_level, risk_reasons = payment_risk_analysis(
        conn, session["user_id"], receiver, amount
    )
    risk_key = payment_risk_key(receiver, amount)
    approved_key = session.get("payment_risk_approved_key")
    approved_at = session.get("payment_risk_approved_at")

    if risk_level == "HIGH" or (risk_level == "MEDIUM" and (
            approved_key != risk_key or not approved_at or time.time() - float(approved_at) > 120)):
        action = "BLOCKED_HIGH_RISK" if risk_level == "HIGH" else "MEDIUM_RISK_REQUIRES_CONFIRMATION"
        conn.execute("""
            INSERT INTO risk_events (user_id, receiver, amount, risk_score, risk_level, reasons, action)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"], receiver, amount, risk_score, risk_level,
            json.dumps(risk_reasons), action
        ))
        conn.commit()
        conn.close()
        session.pop("payment_otp_verified_at", None)
        if risk_level == "HIGH":
            session["risk_block_message"] = "High-risk payment blocked. " + "; ".join(risk_reasons)
        else:
            session["risk_confirmation_required"] = True
            session["risk_pending_key"] = risk_key
            session["risk_pending_score"] = risk_score
        return redirect(url_for("dashboard"))

    conn.execute("""
        INSERT INTO risk_events (user_id, receiver, amount, risk_score, risk_level, reasons, action)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        session["user_id"], receiver, amount, risk_score, risk_level,
        json.dumps(risk_reasons), "PAYMENT_ALLOWED"
    ))

    transaction_id = (
        "TXN-"
        + uuid.uuid4().hex[:12].upper()
    )

    conn.execute("""
        INSERT INTO transactions
        (
            user_id,
            receiver,
            receiver_name,
            amount,
            status,
            transaction_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session["user_id"],
        receiver,
        receiver_user["name"],
        amount,
        "SUCCESS",
        transaction_id
    ))

    sender = conn.execute("""
        SELECT name
        FROM users
        WHERE id = ?
    """, (session["user_id"],)).fetchone()

    sender_name = sender["name"] if sender else "PalmPay User"

    # Keep the customer's existing transaction unchanged. For merchant recipients,
    # also create a separate incoming ledger entry so Merchant Mode can report sales.
    if merchant_profile:
        conn.execute("""
            INSERT INTO transactions
            (user_id, receiver, receiver_name, amount, status, transaction_id)
            VALUES (?, ?, ?, ?, 'RECEIVED', ?)
        """, (
            receiver_user["id"],
            session.get("user_mobile", "Unknown"),
            sender_name,
            amount,
            transaction_id
        ))
        conn.execute("""
            INSERT INTO notifications
            (user_id, title, message, amount, transaction_id, is_read)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (
            receiver_user["id"],
            "Merchant payment received",
            f"{sender_name} paid ₹{amount:.2f} to your business.",
            amount,
            transaction_id
        ))

    conn.execute("""
        INSERT INTO notifications
        (
            user_id,
            title,
            message,
            amount,
            transaction_id,
            is_read
        )
        VALUES (?, ?, ?, ?, ?, 0)
    """, (
        receiver_user["id"],
        "Payment Received",
        f"₹{amount:.2f} received from {sender_name}",
        amount,
        transaction_id
    ))

    conn.commit()
    conn.close()

    # One successful palm verification = one payment authorization.
    session.pop("palm_verified_at", None)
    session.pop("palm_verification_token", None)
    session.pop("payment_pin_verified_at", None)

    return redirect(
        url_for(
            "payment_success",
            transaction_id=transaction_id
        )
    )

# =========================================================
# NOTIFICATIONS API
# =========================================================

@app.route("/api/notifications", methods=["GET"])
def get_notifications():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "Please login first."
        }), 401

    conn = get_db()

    rows = conn.execute("""
        SELECT id, title, message, amount, transaction_id, is_read, created_at
        FROM notifications
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 30
    """, (session["user_id"],)).fetchall()

    unread = conn.execute("""
        SELECT COUNT(*) AS total
        FROM notifications
        WHERE user_id = ? AND is_read = 0
    """, (session["user_id"],)).fetchone()["total"]

    conn.close()

    return jsonify({
        "success": True,
        "unread": unread,
        "notifications": [dict(row) for row in rows]
    })


@app.route("/api/notifications/read", methods=["POST"])
def mark_notification_read():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "Please login first."
        }), 401

    data = request.get_json(silent=True) or {}
    notification_id = data.get("id")

    conn = get_db()

    if notification_id == "all":
        conn.execute("""
            UPDATE notifications
            SET is_read = 1
            WHERE user_id = ?
        """, (session["user_id"],))
    else:
        try:
            notification_id = int(notification_id)
        except (TypeError, ValueError):
            conn.close()
            return jsonify({
                "success": False,
                "message": "Invalid notification id."
            }), 400

        conn.execute("""
            UPDATE notifications
            SET is_read = 1
            WHERE id = ? AND user_id = ?
        """, (notification_id, session["user_id"]))

    conn.commit()

    unread = conn.execute("""
        SELECT COUNT(*) AS total
        FROM notifications
        WHERE user_id = ? AND is_read = 0
    """, (session["user_id"],)).fetchone()["total"]

    conn.close()

    return jsonify({
        "success": True,
        "unread": unread
    })


# =========================================================
# =========================================================
# VIRTUAL PALMPAY CARD + QR PAYMENT
# =========================================================

def generate_unique_card_number(conn):
    while True:
        number = "".join(str(secrets.randbelow(10)) for _ in range(16))
        exists = conn.execute(
            "SELECT id FROM virtual_cards WHERE card_number = ? LIMIT 1",
            (number,)
        ).fetchone()
        if not exists:
            return number


def get_or_create_virtual_card(user_id):
    conn = get_db()

    card = conn.execute("""
        SELECT id, user_id, card_number, expiry, cvv, is_frozen, created_at
        FROM virtual_cards
        WHERE user_id = ?
        LIMIT 1
    """, (user_id,)).fetchone()

    if not card:
        card_number = generate_unique_card_number(conn)
        expiry = time.strftime("%m/%y", time.localtime(time.time() + 365 * 24 * 60 * 60 * 3))
        cvv = f"{secrets.randbelow(1000):03d}"

        conn.execute("""
            INSERT INTO virtual_cards
            (user_id, card_number, expiry, cvv, is_frozen)
            VALUES (?, ?, ?, ?, 0)
        """, (user_id, card_number, expiry, cvv))
        conn.commit()

        card = conn.execute("""
            SELECT id, user_id, card_number, expiry, cvv, is_frozen, created_at
            FROM virtual_cards
            WHERE user_id = ?
            LIMIT 1
        """, (user_id,)).fetchone()

    conn.close()
    return card


@app.route("/api/virtual-card", methods=["GET"])
def virtual_card():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "Please login first."
        }), 401

    card = get_or_create_virtual_card(session["user_id"])

    return jsonify({
        "success": True,
        "card": {
            "card_number": card["card_number"],
            "masked_number": " ".join([
                card["card_number"][0:4],
                "••••",
                "••••",
                card["card_number"][-4:]
            ]),
            "expiry": card["expiry"],
            "cvv": card["cvv"],
            "is_frozen": bool(card["is_frozen"])
        }
    })


@app.route("/api/virtual-card/toggle-freeze", methods=["POST"])
def toggle_virtual_card_freeze():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "Please login first."
        }), 401

    card = get_or_create_virtual_card(session["user_id"])
    new_state = 0 if card["is_frozen"] else 1

    conn = get_db()
    conn.execute("""
        UPDATE virtual_cards
        SET is_frozen = ?
        WHERE user_id = ?
    """, (new_state, session["user_id"]))
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "is_frozen": bool(new_state),
        "message": "Card frozen." if new_state else "Card unfrozen."
    })



# =========================================================
# MERCHANT MODE
# =========================================================

def serialize_merchant_profile(row):
    if not row:
        return None
    return {
        "merchant_id": row["merchant_id"],
        "business_name": row["business_name"],
        "category": row["category"],
        "accepting_payments": bool(row["accepting_payments"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


@app.route("/api/merchant/profile", methods=["GET", "POST"])
def merchant_profile_api():
    if session.get("portal_mode") != "merchant" or "user_id" not in session:
        return jsonify({"success": False, "message": "Please login first."}), 401

    conn = get_db()
    row = conn.execute("""
        SELECT merchant_id, business_name, category, accepting_payments, created_at, updated_at
        FROM merchant_profiles
        WHERE user_id = ?
        LIMIT 1
    """, (session["user_id"],)).fetchone()

    if request.method == "GET":
        conn.close()
        return jsonify({"success": True, "profile": serialize_merchant_profile(row)})

    data = request.get_json(silent=True) or {}
    business_name = str(data.get("business_name", "")).strip()
    category = str(data.get("category", "Other")).strip() or "Other"
    allowed = {"Retail", "Food & Beverage", "Education", "Services", "Freelance", "Other"}
    if not business_name:
        conn.close()
        return jsonify({"success": False, "message": "Business name is required."}), 400
    if len(business_name) > 60:
        conn.close()
        return jsonify({"success": False, "message": "Business name is too long."}), 400
    if category not in allowed:
        category = "Other"

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if row:
        conn.execute("""
            UPDATE merchant_profiles
            SET business_name = ?, category = ?, updated_at = ?
            WHERE user_id = ?
        """, (business_name, category, now, session["user_id"]))
    else:
        merchant_id = "MCH-" + uuid.uuid4().hex[:10].upper()
        conn.execute("""
            INSERT INTO merchant_profiles
            (user_id, merchant_id, business_name, category, accepting_payments, updated_at)
            VALUES (?, ?, ?, ?, 1, ?)
        """, (session["user_id"], merchant_id, business_name, category, now))

    conn.commit()
    row = conn.execute("""
        SELECT merchant_id, business_name, category, accepting_payments, created_at, updated_at
        FROM merchant_profiles
        WHERE user_id = ?
        LIMIT 1
    """, (session["user_id"],)).fetchone()
    conn.close()
    return jsonify({"success": True, "message": "Merchant profile saved successfully.", "profile": serialize_merchant_profile(row)})


@app.route("/api/merchant/accepting", methods=["POST"])
def merchant_accepting_api():
    if session.get("portal_mode") != "merchant" or "user_id" not in session:
        return jsonify({"success": False, "message": "Please login first."}), 401
    data = request.get_json(silent=True) or {}
    accepting = bool(data.get("accepting_payments", False))
    conn = get_db()
    row = conn.execute("SELECT id FROM merchant_profiles WHERE user_id = ?", (session["user_id"],)).fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": "Activate Merchant Mode first."}), 404
    conn.execute("""
        UPDATE merchant_profiles
        SET accepting_payments = ?, updated_at = ?
        WHERE user_id = ?
    """, (1 if accepting else 0, time.strftime("%Y-%m-%d %H:%M:%S"), session["user_id"]))
    conn.commit()
    conn.close()
    log_security_activity(session["user_id"], "Merchant payment status changed", "Accepting customer payments" if accepting else "Customer payments paused")
    return jsonify({"success": True, "accepting_payments": accepting})


@app.route("/api/merchant/overview", methods=["GET"])
def merchant_overview_api():
    if not merchant_auth_required():
        return jsonify({"success": False, "message": "Please login first."}), 401

    conn = get_db()
    merchant_user_id = session["user_id"]
    profile = conn.execute("""
        SELECT merchant_id, business_name, category, accepting_payments, created_at, updated_at
        FROM merchant_profiles
        WHERE user_id = ?
        LIMIT 1
    """, (merchant_user_id,)).fetchone()

    if not profile:
        conn.close()
        return jsonify({"success": False, "message": "Merchant profile not found."}), 404

    # Gross merchant sales are all incoming receipts, including refunded ones.
    gross_row = conn.execute("""
        SELECT COALESCE(SUM(amount),0) AS total, COUNT(*) AS count
        FROM transactions
        WHERE user_id = ? AND status IN ('RECEIVED','PARTIAL_REFUND','REFUNDED')
    """, (merchant_user_id,)).fetchone()
    refund_row = conn.execute("""
        SELECT COALESCE(SUM(refund_amount),0) AS total
        FROM merchant_refunds
        WHERE merchant_user_id = ? AND status = 'COMPLETED'
    """, (merchant_user_id,)).fetchone()

    today_gross = conn.execute("""
        SELECT COALESCE(SUM(amount),0) AS total, COUNT(*) AS count
        FROM transactions
        WHERE user_id = ? AND status IN ('RECEIVED','PARTIAL_REFUND','REFUNDED')
          AND date(created_at,'localtime') = date('now','localtime')
    """, (merchant_user_id,)).fetchone()
    today_refund = conn.execute("""
        SELECT COALESCE(SUM(r.refund_amount),0) AS total
        FROM merchant_refunds r
        JOIN transactions t ON t.transaction_id = r.transaction_id AND t.user_id = r.merchant_user_id
        WHERE r.merchant_user_id = ? AND r.status='COMPLETED'
          AND date(r.created_at,'localtime') = date('now','localtime')
          AND date(t.created_at,'localtime') = date('now','localtime')
    """, (merchant_user_id,)).fetchone()

    unique_row = conn.execute("""
        SELECT COUNT(DISTINCT CASE WHEN receiver IS NOT NULL AND receiver <> 'POS Counter' THEN receiver END) AS customers
        FROM transactions
        WHERE user_id = ? AND status IN ('RECEIVED','PARTIAL_REFUND','REFUNDED')
    """, (merchant_user_id,)).fetchone()

    rows = conn.execute("""
        SELECT t.id, t.receiver AS sender_mobile, t.receiver_name AS sender_name,
               t.amount, t.status, t.transaction_id, t.created_at,
               COALESCE((SELECT SUM(r.refund_amount) FROM merchant_refunds r
                         WHERE r.merchant_user_id = t.user_id
                           AND r.transaction_id = t.transaction_id
                           AND r.status='COMPLETED'),0) AS refunded_amount
        FROM transactions t
        WHERE t.user_id = ? AND t.status IN ('RECEIVED','PARTIAL_REFUND','REFUNDED')
        ORDER BY t.id DESC
        LIMIT 120
    """, (merchant_user_id,)).fetchall()

    trend_rows = conn.execute("""
        SELECT date(t.created_at,'localtime') AS day, COALESCE(SUM(t.amount),0) AS gross
        FROM transactions t
        WHERE t.user_id = ?
          AND t.status IN ('RECEIVED','PARTIAL_REFUND','REFUNDED')
          AND date(t.created_at,'localtime') >= date('now','localtime','-6 day')
        GROUP BY date(t.created_at,'localtime')
        ORDER BY day ASC
    """, (merchant_user_id,)).fetchall()
    trend_map = {r["day"]: float(r["gross"] or 0) for r in trend_rows}

    import datetime as _dt
    today = _dt.date.today()
    sales_by_day=[]
    for i in range(6,-1,-1):
        d=today-_dt.timedelta(days=i)
        day=d.isoformat()
        gross=trend_map.get(day,0.0)
        refund = conn.execute("""
            SELECT COALESCE(SUM(refund_amount),0) AS total
            FROM merchant_refunds
            WHERE merchant_user_id = ? AND status='COMPLETED'
              AND date(created_at,'localtime') = ?
        """, (merchant_user_id, day)).fetchone()["total"]
        sales_by_day.append({"day":day,"total":round(max(0,gross-float(refund or 0)),2),"count":0})

    # Last 24 hourly buckets for a compact smart-analytics graph.
    hourly = []
    now_local = _dt.datetime.now()
    for i in range(23,-1,-1):
        slot=(now_local-_dt.timedelta(hours=i)).replace(minute=0,second=0,microsecond=0)
        label=slot.strftime('%H:00')
        gross = conn.execute("""
            SELECT COALESCE(SUM(amount),0) AS total
            FROM transactions
            WHERE user_id = ? AND status IN ('RECEIVED','PARTIAL_REFUND','REFUNDED')
              AND strftime('%Y-%m-%d %H', datetime(created_at,'localtime')) = ?
        """, (merchant_user_id, slot.strftime('%Y-%m-%d %H'))).fetchone()["total"]
        refund = conn.execute("""
            SELECT COALESCE(SUM(refund_amount),0) AS total
            FROM merchant_refunds
            WHERE merchant_user_id = ? AND status='COMPLETED'
              AND strftime('%Y-%m-%d %H', datetime(created_at,'localtime')) = ?
        """, (merchant_user_id, slot.strftime('%Y-%m-%d %H'))).fetchone()["total"]
        hourly.append({"hour":label,"total":round(max(0,float(gross or 0)-float(refund or 0)),2)})

    top_products = conn.execute("""
        SELECT i.product_name AS name,
               SUM(i.quantity) AS quantity,
               SUM(i.line_total) AS revenue
        FROM merchant_pos_sale_items i
        JOIN merchant_pos_sales s ON s.id = i.pos_sale_id
        WHERE s.merchant_user_id = ? AND s.status='COMPLETED'
        GROUP BY i.product_name
        ORDER BY revenue DESC
        LIMIT 8
    """, (merchant_user_id,)).fetchall()

    product_count = conn.execute("SELECT COUNT(*) AS c FROM merchant_products WHERE merchant_user_id = ? AND active=1", (merchant_user_id,)).fetchone()["c"]
    low_stock_count = conn.execute("SELECT COUNT(*) AS c FROM merchant_products WHERE merchant_user_id = ? AND active=1 AND stock <= 5", (merchant_user_id,)).fetchone()["c"]
    pos_today = conn.execute("""
        SELECT COUNT(*) AS c FROM merchant_pos_sales
        WHERE merchant_user_id = ? AND status='COMPLETED' AND date(created_at,'localtime')=date('now','localtime')
    """, (merchant_user_id,)).fetchone()["c"]
    recent_refunds = conn.execute("""
        SELECT refund_id, transaction_id, refund_amount, reason, created_at
        FROM merchant_refunds WHERE merchant_user_id=? ORDER BY id DESC LIMIT 8
    """, (merchant_user_id,)).fetchall()
    conn.close()

    gross_total=float(gross_row["total"] or 0)
    refund_total=float(refund_row["total"] or 0)
    net_total=max(0,gross_total-refund_total)
    today_net=max(0,float(today_gross["total"] or 0)-float(today_refund["total"] or 0))
    count=int(gross_row["count"] or 0)

    clean_rows=[]
    for row in rows:
        item=dict(row)
        item["amount"]=float(item.get("amount") or 0)
        item["refunded_amount"]=float(item.get("refunded_amount") or 0)
        clean_rows.append(item)

    return jsonify({
        "success":True,
        "profile":serialize_merchant_profile(profile),
        "today_received":today_net,
        "today_gross":float(today_gross["total"] or 0),
        "today_refunds":float(today_refund["total"] or 0),
        "today_payment_count":int(today_gross["count"] or 0),
        "total_received":net_total,
        "gross_received":gross_total,
        "refund_total":refund_total,
        "payment_count":count,
        "average_payment":(gross_total/count) if count else 0,
        "unique_customers":int(unique_row["customers"] or 0),
        "recent_payments":clean_rows,
        "sales_by_day":sales_by_day,
        "hourly_sales":hourly,
        "top_products":[dict(r) for r in top_products],
        "recent_refunds":[dict(r) for r in recent_refunds],
        "product_count":int(product_count or 0),
        "low_stock_count":int(low_stock_count or 0),
        "pos_today":int(pos_today or 0)
    })


@app.route("/api/qr-receiver", methods=["POST"])
def qr_receiver():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Please login first."}), 401

    data = request.get_json(silent=True) or {}
    payload = str(data.get("payload", "")).strip()
    parts = payload.split("|")
    prefix = parts[0].upper() if parts else ""

    if prefix == "PALMPAY":
        if len(parts) != 2:
            return jsonify({"success": False, "message": "Invalid PalmPay QR code."}), 400
        mobile = normalize_indian_mobile(parts[1])
        if not mobile:
            return jsonify({"success": False, "message": "The QR code does not contain a valid Indian mobile number."}), 400
        conn = get_db()
        user = conn.execute("SELECT id, name, mobile FROM users WHERE mobile = ? LIMIT 1", (mobile,)).fetchone()
        conn.close()
        if not user:
            return jsonify({"success": False, "message": "No PalmPay user was found for this QR code."}), 404
        if user["id"] == session["user_id"]:
            return jsonify({"success": False, "message": "This QR code belongs to your own account."}), 400
        return jsonify({"success": True, "name": user["name"], "mobile": user["mobile"], "is_merchant": False})

    if prefix == "PALMPAYM":
        if len(parts) not in (2, 4):
            return jsonify({"success": False, "message": "Invalid merchant QR code."}), 400
        merchant_id = parts[1].strip()
        amount = None
        note = ""
        if len(parts) == 4:
            try:
                amount = float(parts[2])
            except (TypeError, ValueError):
                return jsonify({"success": False, "message": "Invalid merchant QR amount."}), 400
            if not math.isfinite(amount) or amount <= 0:
                return jsonify({"success": False, "message": "Invalid merchant QR amount."}), 400
            note = parts[3]
        conn = get_db()
        merchant = conn.execute("""
            SELECT mp.merchant_id, mp.business_name, mp.accepting_payments,
                   u.id, u.name, u.mobile
            FROM merchant_profiles mp
            JOIN users u ON u.id = mp.user_id
            WHERE mp.merchant_id = ?
            LIMIT 1
        """, (merchant_id,)).fetchone()
        conn.close()
        if not merchant:
            return jsonify({"success": False, "message": "Merchant account not found."}), 404
        if merchant["id"] == session["user_id"]:
            return jsonify({"success": False, "message": "This merchant profile belongs to your own account."}), 400
        if not merchant["accepting_payments"]:
            return jsonify({"success": False, "message": "This merchant is currently not accepting payments."}), 409
        return jsonify({
            "success": True,
            "name": merchant["business_name"],
            "mobile": merchant["mobile"],
            "merchant_id": merchant["merchant_id"],
            "is_merchant": True,
            "amount": amount,
            "note": note
        })

    return jsonify({"success": False, "message": "This is not a valid PalmPay QR code."}), 400



# =========================================================
# FAMILY / MULTI-PALM WALLET
# =========================================================

@app.route("/api/family-pals", methods=["GET", "POST"])
def family_palm_profiles_api():
    if "user_id" not in session or session.get("portal_mode") != "customer":
        return jsonify({"success": False, "message": "Customer login required."}), 401

    if request.method == "GET":
        conn = get_db()
        rows = conn.execute("""
            SELECT id, profile_name, relationship, palm_id, permissions, active, created_at
            FROM palm_profiles
            WHERE user_id = ? AND active = 1 AND relationship != 'Primary'
            ORDER BY id DESC
        """, (session["user_id"],)).fetchall()
        conn.close()
        result = []
        for row in rows:
            try:
                permissions = json.loads(row["permissions"] or '{"payment":true}')
            except Exception:
                permissions = {"payment": True}
            result.append({
                "id": int(row["id"]),
                "profile_name": row["profile_name"],
                "relationship": row["relationship"],
                "palm_id": row["palm_id"],
                "payment_permission": bool(permissions.get("payment", True)),
                "created_at": row["created_at"]
            })
        return jsonify({"success": True, "profiles": result})

    data = request.get_json(silent=True) or {}
    profile_name = str(data.get("profile_name", "")).strip()[:50]
    relationship = str(data.get("relationship", "Family")).strip()[:30] or "Family"
    pin = str(data.get("pin", "")).strip()
    palm_images = data.get("palm_images") or []
    handedness = str(data.get("handedness", "Unknown"))
    challenge = validate_palm_security_challenge(
        str(data.get("challenge_token", "")).strip(),
        data.get("liveness_metrics") or {}
    )

    if len(profile_name) < 2:
        return jsonify({"success": False, "message": "Enter a family profile name."}), 400
    if not re.fullmatch(r"\d{4}", pin):
        return jsonify({"success": False, "message": "Enter your 4-digit Payment PIN to authorize palm enrollment."}), 400
    if not challenge["valid"]:
        return jsonify({"success": False, "message": "Anti-spoof enrollment check failed. Move the palm slowly left and right and try again."}), 403
    if not isinstance(palm_images, list) or len(palm_images) < 4:
        return jsonify({"success": False, "message": "Capture at least 4 palm images."}), 400

    conn = get_db()
    user = conn.execute("SELECT payment_pin FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if not user or not user["payment_pin"] or not check_password_hash(user["payment_pin"], pin):
        conn.close()
        return jsonify({"success": False, "message": "Incorrect Payment PIN."}), 403

    try:
        embeddings = [ccnet_embedding_from_image(x) for x in palm_images[:8]]
        template = average_ccnet_embeddings(embeddings)
    except Exception as exc:
        conn.close()
        return jsonify({"success": False, "message": f"Family palm enrollment failed: {exc}"}), 500

    palm_id = "PALM-FAM-" + uuid.uuid4().hex[:10].upper()
    payload = {
        "version": 3,
        "method": "CCNet_2048D",
        "template": None,
        "ccnet_embedding": template,
        "ccnet_samples": len(embeddings),
        "handedness": handedness
    }
    conn.execute("""
        INSERT INTO palm_profiles
        (user_id, profile_name, relationship, palm_id, palm_template, permissions, active)
        VALUES (?, ?, ?, ?, ?, ?, 1)
    """, (
        session["user_id"], profile_name, relationship, palm_id,
        json.dumps(payload), json.dumps({"payment": bool(data.get("payment_permission", True))})
    ))
    conn.commit()
    conn.close()

    log_security_activity(session["user_id"], "Family palm registered", f"{profile_name} · {relationship} · anti-spoof risk {challenge['spoof_score']}")
    return jsonify({
        "success": True,
        "message": "Family palm registered successfully.",
        "palm_id": palm_id,
        "anti_spoof": challenge
    })


@app.route("/api/family-pals/<int:profile_id>", methods=["DELETE", "PUT"])
def family_palm_profile_detail_api(profile_id):
    if "user_id" not in session or session.get("portal_mode") != "customer":
        return jsonify({"success": False, "message": "Customer login required."}), 401

    conn = get_db()
    row = conn.execute("SELECT id, profile_name, permissions FROM palm_profiles WHERE id = ? AND user_id = ? AND relationship != 'Primary' AND active = 1", (profile_id, session["user_id"])).fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": "Family palm profile not found."}), 404

    if request.method == "DELETE":
        conn.execute("UPDATE palm_profiles SET active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?", (profile_id, session["user_id"]))
        conn.commit()
        conn.close()
        log_security_activity(session["user_id"], "Family palm removed", row["profile_name"])
        return jsonify({"success": True, "message": "Family palm profile removed."})

    data = request.get_json(silent=True) or {}
    payment_permission = bool(data.get("payment_permission", True))
    conn.execute("UPDATE palm_profiles SET permissions = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?", (json.dumps({"payment": payment_permission}), profile_id, session["user_id"]))
    conn.commit()
    conn.close()
    log_security_activity(session["user_id"], "Family palm permission changed", f"{row['profile_name']} · payment={payment_permission}")
    return jsonify({"success": True, "message": "Family palm permission updated."})


# =========================================================
# SECURITY COMMAND CENTER
# =========================================================

@app.route("/security-center")
def security_center():
    if "user_id" not in session or session.get("portal_mode") != "customer":
        return redirect(url_for("login"))
    return render_template("security_center.html", name=session.get("user_name", "Customer"))


@app.route("/api/security/command-center")
def security_command_center_api():
    if "user_id" not in session or session.get("portal_mode") != "customer":
        return jsonify({"success": False, "message": "Customer login required."}), 401

    conn = get_db()
    user_id = session["user_id"]
    risk = conn.execute("SELECT COUNT(*) c FROM risk_events WHERE user_id=? AND datetime(created_at)>=datetime('now','-24 hours')", (user_id,)).fetchone()["c"]
    high = conn.execute("SELECT COUNT(*) c FROM risk_events WHERE user_id=? AND risk_level='HIGH' AND datetime(created_at)>=datetime('now','-24 hours')", (user_id,)).fetchone()["c"]
    spoof_fails = conn.execute("SELECT COUNT(*) c FROM palm_security_events WHERE user_id=? AND event_type='ANTI_SPOOF_FAIL' AND datetime(created_at)>=datetime('now','-24 hours')", (user_id,)).fetchone()["c"]
    palm_success = conn.execute("SELECT COUNT(*) c FROM palm_security_events WHERE user_id=? AND event_type='PALM_MATCH_SUCCESS' AND datetime(created_at)>=datetime('now','-24 hours')", (user_id,)).fetchone()["c"]
    events = conn.execute("SELECT event_type, score, details, created_at FROM palm_security_events WHERE user_id=? ORDER BY id DESC LIMIT 15", (user_id,)).fetchall()
    conn.close()
    last = session.get("matched_palm_profile_name")
    return jsonify({
        "success": True,
        "risk_events_24h": int(risk),
        "high_risk_24h": int(high),
        "anti_spoof_failures_24h": int(spoof_fails),
        "palm_success_24h": int(palm_success),
        "last_matched_profile": last,
        "events": [dict(x) for x in events]
    })


# =========================================================
# RECEIPT VERIFICATION
# =========================================================

@app.route("/verify-receipt/<transaction_id>")
def verify_receipt(transaction_id):
    conn = get_db()
    row = conn.execute("""
        SELECT transaction_id, receiver, receiver_name, amount, status, created_at
        FROM transactions WHERE transaction_id=? LIMIT 1
    """, (transaction_id,)).fetchone()
    conn.close()
    if not row:
        return render_template("receipt_verify.html", transaction=None), 404
    return render_template("receipt_verify.html", transaction=row)


@app.route("/api/receipt/<transaction_id>")
def receipt_api(transaction_id):
    conn = get_db()
    row = conn.execute("SELECT transaction_id, receiver, receiver_name, amount, status, created_at FROM transactions WHERE transaction_id=? LIMIT 1", (transaction_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"success": False, "message": "Receipt not found."}), 404
    return jsonify({"success": True, "transaction": dict(row)})


# =========================================================
# ADMIN CONTROL CENTER
# =========================================================

ADMIN_EMAIL = os.getenv("PALMPAY_ADMIN_EMAIL", "admin@palmpay.local").strip().lower()
ADMIN_PASSWORD = os.getenv("PALMPAY_ADMIN_PASSWORD", "")


def admin_auth_required():
    return session.get("portal_mode") == "admin" and bool(session.get("admin_email"))


def find_admin_account(email):
    conn = get_db()
    row = conn.execute("""
        SELECT id, name, email, password, role, is_active
        FROM admin_accounts
        WHERE lower(email) = ?
        LIMIT 1
    """, (email.strip().lower(),)).fetchone()
    conn.close()
    return row


@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # First try registered admin accounts.
        account = find_admin_account(email) if email else None
        if account and int(account["is_active"] or 0) == 1:
            try:
                valid_password = check_password_hash(account["password"], password)
            except Exception:
                valid_password = False
            if valid_password:
                session.clear()
                session["portal_mode"] = "admin"
                session["admin_id"] = int(account["id"])
                session["admin_email"] = account["email"]
                session["admin_name"] = account["name"]
                session["admin_role"] = account["role"] or "Administrator"
                record_admin_audit("Admin login")
                return redirect(url_for("admin_dashboard"))

        # Enable the first-install administrator only when explicitly configured.
        if ADMIN_PASSWORD and email == ADMIN_EMAIL and secrets.compare_digest(password, ADMIN_PASSWORD):
            session.clear()
            session["portal_mode"] = "admin"
            session["admin_id"] = 0
            session["admin_email"] = ADMIN_EMAIL
            session["admin_name"] = "System Administrator"
            session["admin_role"] = "System Administrator"
            record_admin_audit("Admin login", details="Default demo administrator")
            return redirect(url_for("admin_dashboard"))

        return render_template(
            "admin_login.html",
            error="Invalid administrator credentials.",
            admin_email=ADMIN_EMAIL
        )

    return render_template("admin_login.html", admin_email=ADMIN_EMAIL)


@app.route("/admin-register", methods=["GET", "POST"])
def admin_register():
    # Registration is intentionally reachable from the admin login page.
    # Existing admins remain unaffected.
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(name) < 2:
            return render_template("admin_register.html", error="Please enter a valid administrator name.")
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            return render_template("admin_register.html", error="Please enter a valid administrator email.")
        if len(password) < 8:
            return render_template("admin_register.html", error="Password must be at least 8 characters.")
        if password != confirm_password:
            return render_template("admin_register.html", error="Passwords do not match.")

        # Do not allow a registration to silently overwrite an existing account.
        if ADMIN_PASSWORD and email == ADMIN_EMAIL:
            return render_template("admin_register.html", error="That email is reserved for the default administrator.")

        conn = get_db()
        existing = conn.execute(
            "SELECT id FROM admin_accounts WHERE lower(email)=? LIMIT 1",
            (email,)
        ).fetchone()
        if existing:
            conn.close()
            return render_template("admin_register.html", error="An administrator account with this email already exists.")

        password_hash = generate_password_hash(password)
        try:
            cur = conn.execute("""
                INSERT INTO admin_accounts (name, email, password, role, is_active)
                VALUES (?, ?, ?, 'Administrator', 1)
            """, (name, email, password_hash))
            conn.commit()
            admin_id = cur.lastrowid
        except sqlite3.IntegrityError:
            conn.rollback()
            conn.close()
            return render_template("admin_register.html", error="Unable to create account. The email may already be registered.")
        conn.close()

        session.clear()
        session["portal_mode"] = "admin"
        session["admin_id"] = int(admin_id)
        session["admin_email"] = email
        session["admin_name"] = name
        session["admin_role"] = "Administrator"
        record_admin_audit("Admin account registered", email, f"Created administrator account for {name}")
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_register.html")


@app.route("/admin/dashboard")
def admin_dashboard():
    if not admin_auth_required():
        return redirect(url_for("admin_login"))
    return render_template(
        "admin_dashboard.html",
        admin_email=session.get("admin_email", ADMIN_EMAIL),
        admin_name=session.get("admin_name", "Administrator"),
        admin_role=session.get("admin_role", "Administrator")
    )


@app.route("/api/admin/overview")
def admin_overview_api():
    if not admin_auth_required():
        return jsonify({"success": False, "message": "Administrator login required."}), 401
    conn = get_db()
    stats = {
        "users": conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"],
        "active_users": conn.execute("SELECT COUNT(*) c FROM users WHERE account_status='ACTIVE'").fetchone()["c"],
        "suspended_users": conn.execute("SELECT COUNT(*) c FROM users WHERE account_status='SUSPENDED'").fetchone()["c"],
        "merchants": conn.execute("SELECT COUNT(*) c FROM merchant_profiles").fetchone()["c"],
        "transactions": conn.execute("SELECT COUNT(*) c FROM transactions").fetchone()["c"],
        "completed_payments": conn.execute("SELECT COUNT(*) c FROM transactions WHERE status IN ('SUCCESS','RECEIVED','PARTIAL_REFUND')").fetchone()["c"],
        "high_risk": conn.execute("SELECT COUNT(*) c FROM risk_events WHERE risk_level='HIGH' AND datetime(created_at)>=datetime('now','-24 hours')").fetchone()["c"],
        "refunds": conn.execute("SELECT COALESCE(SUM(refund_amount),0) total FROM merchant_refunds WHERE status='COMPLETED'").fetchone()["total"],
        "pos_sales": conn.execute("SELECT COALESCE(SUM(amount),0) total FROM merchant_pos_sales WHERE status='COMPLETED'").fetchone()["total"],
        "spoof_fails": conn.execute("SELECT COUNT(*) c FROM palm_security_events WHERE event_type='ANTI_SPOOF_FAIL' AND datetime(created_at)>=datetime('now','-24 hours')").fetchone()["c"],
    }
    users = conn.execute("SELECT id,name,mobile,email,account_status,created_at FROM users ORDER BY id DESC LIMIT 25").fetchall()
    merchants = conn.execute("""
        SELECT u.id, u.name, u.email, u.account_status, mp.merchant_id, mp.business_name, mp.category, mp.accepting_payments
        FROM users u JOIN merchant_profiles mp ON mp.user_id=u.id ORDER BY mp.id DESC LIMIT 20
    """).fetchall()
    transactions = conn.execute("""
        SELECT t.transaction_id,t.user_id,t.receiver,t.receiver_name,t.amount,t.status,t.created_at,u.name AS owner_name
        FROM transactions t LEFT JOIN users u ON u.id=t.user_id ORDER BY t.id DESC LIMIT 25
    """).fetchall()
    security = conn.execute("""
        SELECT p.event_type,p.score,p.details,p.created_at,u.name AS owner_name
        FROM palm_security_events p LEFT JOIN users u ON u.id=p.user_id ORDER BY p.id DESC LIMIT 20
    """).fetchall()
    audit = conn.execute("SELECT admin_ref,action,target,details,created_at FROM admin_audit ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()
    return jsonify({
        "success": True,
        "stats": {k: (float(v) if k in {"refunds","pos_sales"} else int(v)) for k,v in stats.items()},
        "users": [dict(x) for x in users],
        "merchants": [dict(x) for x in merchants],
        "transactions": [dict(x) for x in transactions],
        "security": [dict(x) for x in security],
        "audit": [dict(x) for x in audit],
    })


@app.route("/api/admin/users/<int:user_id>/status", methods=["POST"])
def admin_user_status(user_id):
    if not admin_auth_required():
        return jsonify({"success": False, "message": "Administrator login required."}), 401
    data = request.get_json(silent=True) or {}
    status = str(data.get("status", "")).upper()
    if status not in {"ACTIVE", "SUSPENDED"}:
        return jsonify({"success": False, "message": "Invalid account status."}), 400
    conn = get_db()
    row = conn.execute("SELECT name,email FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": "User not found."}), 404
    conn.execute("UPDATE users SET account_status=? WHERE id=?", (status, user_id))
    conn.commit(); conn.close()
    record_admin_audit("Account status changed", str(user_id), f"{row['email']} → {status}")
    return jsonify({"success": True, "message": f"Account set to {status}."})


@app.route("/api/admin/merchants/<int:user_id>/accepting", methods=["POST"])
def admin_merchant_accepting(user_id):
    if not admin_auth_required():
        return jsonify({"success": False, "message": "Administrator login required."}), 401
    data = request.get_json(silent=True) or {}
    accepting = 1 if bool(data.get("accepting_payments", True)) else 0
    conn = get_db()
    row = conn.execute("SELECT merchant_id,business_name FROM merchant_profiles WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": "Merchant not found."}), 404
    conn.execute("UPDATE merchant_profiles SET accepting_payments=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?", (accepting, user_id))
    conn.commit(); conn.close()
    record_admin_audit("Merchant payment status changed", row["merchant_id"], f"{row['business_name']} → {'ACCEPTING' if accepting else 'PAUSED'}")
    return jsonify({"success": True, "message": "Merchant payment status updated."})


# =========================================================
# ADVANCED MERCHANT FEATURES
# =========================================================

def merchant_auth_required():
    return session.get("portal_mode") == "merchant" and "user_id" in session


def serialize_merchant_product(row):
    if not row:
        return None
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "sku": row["sku"] or "",
        "category": row["category"] or "Other",
        "price": float(row["price"] or 0),
        "stock": int(row["stock"] or 0),
        "active": bool(row["active"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


@app.route("/api/merchant/products", methods=["GET", "POST"])
def merchant_products_api():
    if not merchant_auth_required():
        return jsonify({"success": False, "message": "Merchant login required."}), 401

    merchant_user_id = session["user_id"]
    conn = get_db()

    if request.method == "GET":
        rows = conn.execute("""
            SELECT id, name, sku, category, price, stock, active, created_at, updated_at
            FROM merchant_products
            WHERE merchant_user_id = ?
            ORDER BY active DESC, id DESC
        """, (merchant_user_id,)).fetchall()
        conn.close()
        return jsonify({"success": True, "products": [serialize_merchant_product(r) for r in rows]})

    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    sku = str(data.get("sku", "")).strip()[:40]
    category = str(data.get("category", "Other")).strip() or "Other"
    try:
        price = float(data.get("price", 0))
        stock = int(data.get("stock", 0))
    except (TypeError, ValueError):
        conn.close()
        return jsonify({"success": False, "message": "Price and stock must be valid numbers."}), 400

    if len(name) < 2 or len(name) > 80:
        conn.close()
        return jsonify({"success": False, "message": "Product name must be 2-80 characters."}), 400
    if not math.isfinite(price) or price <= 0:
        conn.close()
        return jsonify({"success": False, "message": "Product price must be greater than zero."}), 400
    if stock < 0 or stock > 100000:
        conn.close()
        return jsonify({"success": False, "message": "Stock must be between 0 and 100000."}), 400

    try:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.execute("""
            INSERT INTO merchant_products
            (merchant_user_id, name, sku, category, price, stock, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
        """, (merchant_user_id, name, sku or None, category, price, stock, now, now))
        conn.commit()
        row = conn.execute("""
            SELECT id, name, sku, category, price, stock, active, created_at, updated_at
            FROM merchant_products WHERE id = ?
        """, (cursor.lastrowid,)).fetchone()
    except sqlite3.IntegrityError:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": "Unable to add this product. Check the SKU."}), 409
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return jsonify({"success": True, "product": serialize_merchant_product(row)})


@app.route("/api/merchant/products/<int:product_id>", methods=["PUT", "DELETE"])
def merchant_product_detail_api(product_id):
    if not merchant_auth_required():
        return jsonify({"success": False, "message": "Merchant login required."}), 401

    conn = get_db()
    existing = conn.execute("""
        SELECT id, name, sku, category, price, stock, active, created_at, updated_at
        FROM merchant_products
        WHERE id = ? AND merchant_user_id = ?
    """, (product_id, session["user_id"])).fetchone()
    if not existing:
        conn.close()
        return jsonify({"success": False, "message": "Product not found."}), 404

    if request.method == "DELETE":
        conn.execute("UPDATE merchant_products SET active = 0, updated_at = ? WHERE id = ? AND merchant_user_id = ?",
                     (time.strftime("%Y-%m-%d %H:%M:%S"), product_id, session["user_id"]))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Product archived."})

    data = request.get_json(silent=True) or {}
    name = str(data.get("name", existing["name"])).strip()
    sku = str(data.get("sku", existing["sku"] or "")).strip()[:40]
    category = str(data.get("category", existing["category"] or "Other")).strip() or "Other"
    try:
        price = float(data.get("price", existing["price"]))
        stock = int(data.get("stock", existing["stock"]))
        active = 1 if bool(data.get("active", existing["active"])) else 0
    except (TypeError, ValueError):
        conn.close()
        return jsonify({"success": False, "message": "Invalid product values."}), 400

    if not 2 <= len(name) <= 80 or not math.isfinite(price) or price <= 0 or stock < 0 or stock > 100000:
        conn.close()
        return jsonify({"success": False, "message": "Check the product name, price and stock values."}), 400

    try:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("""
            UPDATE merchant_products
            SET name = ?, sku = ?, category = ?, price = ?, stock = ?, active = ?, updated_at = ?
            WHERE id = ? AND merchant_user_id = ?
        """, (name, sku or None, category, price, stock, active, now, product_id, session["user_id"]))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": "That SKU is already used by another product."}), 409

    row = conn.execute("""
        SELECT id, name, sku, category, price, stock, active, created_at, updated_at
        FROM merchant_products WHERE id = ?
    """, (product_id,)).fetchone()
    conn.close()
    return jsonify({"success": True, "product": serialize_merchant_product(row)})


@app.route("/api/merchant/pos/complete", methods=["POST"])
def merchant_pos_complete_api():
    if not merchant_auth_required():
        return jsonify({"success": False, "message": "Merchant login required."}), 401

    data = request.get_json(silent=True) or {}
    items = data.get("items") or []
    customer_mobile = normalize_indian_mobile(data.get("customer_mobile", ""))
    customer_name = str(data.get("customer_name", "")).strip()[:80] or "Walk-in Customer"

    if not isinstance(items, list) or not items:
        return jsonify({"success": False, "message": "Add at least one item to the POS cart."}), 400
    if len(items) > 50:
        return jsonify({"success": False, "message": "POS cart has too many line items."}), 400

    conn = get_db()
    normalized_items = []
    total = 0.0

    try:
        for raw in items:
            try:
                product_id = int(raw.get("product_id"))
                quantity = int(raw.get("quantity", 0))
            except (TypeError, ValueError, AttributeError):
                raise ValueError("Invalid POS item.")
            if quantity <= 0 or quantity > 1000:
                raise ValueError("Quantity must be between 1 and 1000.")

            row = conn.execute("""
                SELECT id, name, price, stock, active
                FROM merchant_products
                WHERE id = ? AND merchant_user_id = ?
                LIMIT 1
            """, (product_id, session["user_id"])).fetchone()
            if not row or not row["active"]:
                raise ValueError("One of the selected products is not active.")
            if int(row["stock"]) < quantity:
                raise ValueError(f"Not enough stock for {row['name']}. Available: {row['stock']}.")

            line_total = float(row["price"]) * quantity
            total += line_total
            normalized_items.append({
                "product_id": int(row["id"]),
                "product_name": row["name"],
                "quantity": quantity,
                "unit_price": float(row["price"]),
                "line_total": round(line_total, 2)
            })

        if total <= 0 or not math.isfinite(total):
            raise ValueError("Invalid POS total.")

        order_id = "POS-" + uuid.uuid4().hex[:10].upper()
        transaction_id = "TXN-" + uuid.uuid4().hex[:12].upper()
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        sale_cursor = conn.execute("""
            INSERT INTO merchant_pos_sales
            (merchant_user_id, order_id, customer_mobile, customer_name, amount, status, transaction_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'COMPLETED', ?, ?, ?)
        """, (session["user_id"], order_id, customer_mobile, customer_name, round(total, 2), transaction_id, now, now))
        sale_id = sale_cursor.lastrowid

        for item in normalized_items:
            conn.execute("""
                INSERT INTO merchant_pos_sale_items
                (pos_sale_id, product_id, product_name, quantity, unit_price, line_total)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sale_id, item["product_id"], item["product_name"], item["quantity"], item["unit_price"], item["line_total"]))
            conn.execute("""
                UPDATE merchant_products
                SET stock = stock - ?, updated_at = ?
                WHERE id = ? AND merchant_user_id = ?
            """, (item["quantity"], now, item["product_id"], session["user_id"]))

        conn.execute("""
            INSERT INTO transactions
            (user_id, receiver, receiver_name, amount, status, transaction_id)
            VALUES (?, ?, ?, ?, 'RECEIVED', ?)
        """, (session["user_id"], customer_mobile or "POS Counter", customer_name, round(total, 2), transaction_id))

        conn.commit()
    except ValueError as exc:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": str(exc)}), 400
    except sqlite3.Error as exc:
        conn.rollback()
        conn.close()
        app.logger.exception("POS completion failed")
        return jsonify({"success": False, "message": f"POS database error: {exc}"}), 500

    if customer_mobile:
        customer = conn.execute("SELECT id FROM users WHERE mobile = ? LIMIT 1", (customer_mobile,)).fetchone()
        if customer:
            conn.execute("""
                INSERT INTO notifications
                (user_id, title, message, amount, transaction_id, is_read)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (customer["id"], "POS payment recorded", f"₹{total:.2f} paid at {session.get('user_name', 'merchant')}.", round(total, 2), transaction_id))
            conn.commit()
    conn.close()

    log_security_activity(session["user_id"], "POS sale completed", f"{order_id} for ₹{total:.2f}")
    return jsonify({
        "success": True,
        "order_id": order_id,
        "transaction_id": transaction_id,
        "amount": round(total, 2),
        "items": normalized_items,
        "message": "Demo POS sale completed and inventory updated."
    })


@app.route("/api/merchant/pos/orders", methods=["GET"])
def merchant_pos_orders_api():
    if not merchant_auth_required():
        return jsonify({"success": False, "message": "Merchant login required."}), 401
    conn = get_db()
    rows = conn.execute("""
        SELECT order_id, customer_mobile, customer_name, amount, status, transaction_id, created_at
        FROM merchant_pos_sales
        WHERE merchant_user_id = ?
        ORDER BY id DESC LIMIT 30
    """, (session["user_id"],)).fetchall()
    conn.close()
    return jsonify({"success": True, "orders": [dict(r) for r in rows]})


@app.route("/api/merchant/refunds", methods=["GET"])
def merchant_refunds_api():
    if not merchant_auth_required():
        return jsonify({"success": False, "message": "Merchant login required."}), 401
    conn = get_db()
    rows = conn.execute("""
        SELECT refund_id, transaction_id, original_amount, refund_amount, reason, status, created_at
        FROM merchant_refunds
        WHERE merchant_user_id = ?
        ORDER BY id DESC LIMIT 50
    """, (session["user_id"],)).fetchall()
    conn.close()
    return jsonify({"success": True, "refunds": [dict(r) for r in rows]})


@app.route("/api/merchant/refunds", methods=["POST"])
def merchant_create_refund_api():
    if not merchant_auth_required():
        return jsonify({"success": False, "message": "Merchant login required."}), 401

    data = request.get_json(silent=True) or {}
    transaction_id = str(data.get("transaction_id", "")).strip()
    reason = str(data.get("reason", "Customer refund")).strip()[:160] or "Customer refund"
    try:
        refund_amount = float(data.get("amount", 0))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Enter a valid refund amount."}), 400

    if not transaction_id or not math.isfinite(refund_amount) or refund_amount <= 0:
        return jsonify({"success": False, "message": "Invalid refund details."}), 400

    conn = get_db()
    tx = conn.execute("""
        SELECT id, amount, status, receiver, receiver_name
        FROM transactions
        WHERE transaction_id = ? AND user_id = ?
        LIMIT 1
    """, (transaction_id, session["user_id"])).fetchone()
    if not tx:
        conn.close()
        return jsonify({"success": False, "message": "Merchant transaction not found."}), 404

    if tx["status"] not in ("RECEIVED", "PARTIAL_REFUND"):
        conn.close()
        return jsonify({"success": False, "message": "This payment is not eligible for another refund."}), 409

    refunded_row = conn.execute("""
        SELECT COALESCE(SUM(refund_amount), 0) AS total
        FROM merchant_refunds
        WHERE merchant_user_id = ? AND transaction_id = ? AND status = 'COMPLETED'
    """, (session["user_id"], transaction_id)).fetchone()
    already_refunded = float(refunded_row["total"] or 0)
    original_amount = float(tx["amount"] or 0)
    remaining = original_amount - already_refunded

    if refund_amount > remaining + 1e-9:
        conn.close()
        return jsonify({"success": False, "message": f"Maximum refundable amount is ₹{remaining:.2f}."}), 400

    refund_id = "REF-" + uuid.uuid4().hex[:12].upper()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO merchant_refunds
        (merchant_user_id, transaction_id, refund_id, original_amount, refund_amount, reason, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 'COMPLETED', ?)
    """, (session["user_id"], transaction_id, refund_id, original_amount, round(refund_amount, 2), reason, now))

    new_total_refunded = already_refunded + refund_amount
    new_status = "REFUNDED" if new_total_refunded >= original_amount - 1e-9 else "PARTIAL_REFUND"
    conn.execute("UPDATE transactions SET status = ? WHERE id = ? AND user_id = ?", (new_status, tx["id"], session["user_id"]))
    conn.commit()
    conn.close()

    log_security_activity(session["user_id"], "Merchant refund issued", f"{refund_id} · {transaction_id} · ₹{refund_amount:.2f}")
    return jsonify({
        "success": True,
        "refund_id": refund_id,
        "transaction_id": transaction_id,
        "refund_amount": round(refund_amount, 2),
        "remaining": round(max(0, original_amount - new_total_refunded), 2),
        "status": new_status,
        "message": "Refund recorded successfully."
    })


# PALMPAY ASSISTANT / SUPPORT CHAT API
# =========================================================

@app.route("/api/chat", methods=["POST"])
def palm_pay_chat():

    if "user_id" not in session:
        return jsonify({
            "success": False,
            "message": "Please login first."
        }), 401

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "success": False,
            "message": "Please enter a message."
        }), 400

    # Keep the support assistant intentionally local and deterministic.
    # It uses only the logged-in user's own PalmPay database records.
    normalized = re.sub(r"\s+", " ", message.lower()).strip()

    conn = get_db()

    user = conn.execute("""
        SELECT id, name, mobile, email, palm_id
        FROM users
        WHERE id = ?
    """, (session["user_id"],)).fetchone()

    if not user:
        conn.close()
        return jsonify({
            "success": False,
            "message": "User account was not found."
        }), 404

    spent = conn.execute("""
        SELECT COALESCE(SUM(amount), 0) AS total
        FROM transactions
        WHERE user_id = ? AND status = 'SUCCESS'
    """, (user["id"],)).fetchone()["total"]

    added = conn.execute("""
        SELECT COALESCE(SUM(ABS(amount)), 0) AS total
        FROM transactions
        WHERE user_id = ? AND status = 'TOPUP'
    """, (user["id"],)).fetchone()["total"]

    balance = 10000 + added - spent

    unread = conn.execute("""
        SELECT COUNT(*) AS total
        FROM notifications
        WHERE user_id = ? AND is_read = 0
    """, (user["id"],)).fetchone()["total"]

    latest = conn.execute("""
        SELECT receiver, receiver_name, amount, status,
               transaction_id, created_at
        FROM transactions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (user["id"],)).fetchone()

    # A transaction ID can be supplied directly, e.g. "TXN-ABC123..."
    txn_match = re.search(r"\b(?:TXN|TOPUP)-[A-Z0-9]{6,20}\b", message.upper())
    transaction = None

    if txn_match:
        transaction = conn.execute("""
            SELECT receiver, receiver_name, amount, status,
                   transaction_id, created_at
            FROM transactions
            WHERE user_id = ? AND transaction_id = ?
            LIMIT 1
        """, (user["id"], txn_match.group(0))).fetchone()

    conn.close()

    def money(value):
        return f"₹{float(value):,.2f}"

    # Transaction-specific lookup gets priority over generic payment questions.
    if txn_match:
        if not transaction:
            reply = (
                f"I couldn't find {txn_match.group(0)} in your transaction history. "
                "Please check the transaction ID and try again."
            )
        else:
            if transaction["status"] == "TOPUP":
                target = "Wallet Top Up"
            else:
                target = transaction["receiver_name"] or transaction["receiver"]

            reply = (
                f"Transaction {transaction['transaction_id']} is {transaction['status']}. "
                f"Amount: {money(abs(transaction['amount']))}. "
                f"To: {target}. "
                f"Time: {transaction['created_at']}."
            )

    elif any(word in normalized for word in [
        "balance", "how much money", "wallet amount", "wallet balance", "available money"
    ]):
        reply = (
            f"Your current PalmPay demo wallet balance is {money(balance)}. "
            f"You have added {money(added)} and successfully spent {money(spent)}."
        )

    elif any(phrase in normalized for phrase in [
        "where is my payment", "payment status", "latest payment",
        "last payment", "did my payment", "payment completed", "payment complete",
        "recent payment"
    ]):
        if not latest:
            reply = "You don't have any transactions in your PalmPay history yet."
        else:
            if latest["status"] == "TOPUP":
                target = "Wallet Top Up"
            else:
                target = latest["receiver_name"] or latest["receiver"]

            reply = (
                f"Your latest transaction is {latest['status']}. "
                f"Amount: {money(abs(latest['amount']))}. "
                f"Destination: {target}. "
                f"Transaction ID: {latest['transaction_id']}."
            )

    elif any(word in normalized for word in [
        "transaction", "transactions", "history", "payment history", "recent payments"
    ]):
        if not latest:
            reply = "Your transaction history is empty."
        else:
            reply = (
                f"Your latest transaction is {latest['transaction_id']} with status "
                f"{latest['status']} for {money(abs(latest['amount']))}. "
                "You can view the full history in the Transactions section below."
            )

    elif any(word in normalized for word in [
        "notification", "notifications", "alert", "alerts"
    ]):
        if unread:
            reply = f"You have {unread} unread notification{'s' if unread != 1 else ''}. Open the bell icon to view them."
        else:
            reply = "You have no unread notifications right now."

    elif any(word in normalized for word in [
        "palm", "scan", "scanner", "authenticate", "authentication", "biometric"
    ]):
        reply = (
            "Palm authentication uses your registered palm template in this demo. "
            "Open the payment authentication window, allow camera access, keep your palm inside the frame, "
            "and follow the on-screen liveness/gesture checks."
        )

    elif any(word in normalized for word in [
        "send money", "make payment", "pay someone", "payment", "transfer"
    ]):
        reply = (
            "To send money, enter the receiver's registered Indian mobile number and amount. "
            "After the receiver is found, use Authenticate & Pay and complete the palm scan."
        )

    elif any(word in normalized for word in [
        "add money", "top up", "topup", "deposit", "recharge wallet"
    ]):
        reply = (
            "Use the Add Money section to add funds to the demo wallet. "
            "The added amount will appear in your balance and transaction history."
        )

    elif any(word in normalized for word in [
        "help", "what can you do", "commands", "options", "support"
    ]):
        reply = (
            "I can help with balance, payment status, transaction history, notifications, "
            "palm authentication, sending money, and adding money. "
            "Try asking: 'What is my balance?' or 'Where is my payment?'"
        )

    elif any(word in normalized for word in [
        "hello", "hi", "hey", "hai", "good morning", "good evening", "good afternoon"
    ]):
        reply = (
            f"Hi {user['name']}! I'm your PalmPay Assistant. "
            "Ask me about your balance, payments, transactions, notifications, or palm authentication."
        )

    elif any(word in normalized for word in [
        "thank", "thanks", "thank you"
    ]):
        reply = "You're welcome! I'm here whenever you need help with PalmPay."

    else:
        reply = (
            "I can help with your PalmPay account. Try one of these: "
            "'What is my balance?', 'Where is my payment?', 'Show my transactions', "
            "'Any notifications?', or 'How does palm authentication work?'"
        )

    return jsonify({
        "success": True,
        "reply": reply,
        "quick_replies": [
            "What is my balance?",
            "Where is my payment?",
            "Show my transactions",
            "Any notifications?"
        ]
    })


# =========================================================
# PAYMENT SUCCESS
# =========================================================

@app.route("/payment-success/<transaction_id>")
def payment_success(transaction_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    transaction = conn.execute("""
        SELECT *
        FROM transactions
        WHERE transaction_id = ?
        AND user_id = ?
    """, (
        transaction_id,
        session["user_id"]
    )).fetchone()

    conn.close()

    if not transaction:
        return redirect(url_for("dashboard"))

    return render_template(
        "payment_success.html",
        transaction=transaction
    )


# =========================================================
# FRAUD / RISK APIs
# =========================================================

@app.route("/api/payment-risk", methods=["POST"])
def payment_risk_api():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401

    if not session.get("payment_otp_verified_at"):
        return jsonify({"success": False, "message": "Complete OTP verification first."}), 403

    data = request.get_json(silent=True) or {}
    receiver = normalize_indian_mobile(str(data.get("receiver", "")).strip())
    try:
        amount = float(data.get("amount", 0))
    except (TypeError, ValueError):
        amount = 0

    if not receiver or not math.isfinite(amount) or amount <= 0:
        return jsonify({"success": False, "message": "Invalid payment details."}), 400

    conn = get_db()
    receiver_user = conn.execute("SELECT id FROM users WHERE mobile = ?", (receiver,)).fetchone()
    if not receiver_user:
        conn.close()
        return jsonify({"success": False, "message": "Receiver account not found."}), 404

    score, level, reasons = payment_risk_analysis(conn, session["user_id"], receiver, amount)
    conn.close()

    if level == "HIGH":
        message = "This payment has been blocked because the risk score is high."
        requires_confirmation = False
    elif level == "MEDIUM":
        message = "This payment needs one extra confirmation because unusual activity was detected."
        requires_confirmation = True
    else:
        message = "No unusual activity detected."
        requires_confirmation = False

    return jsonify({
        "success": True,
        "score": score,
        "level": level,
        "reasons": reasons,
        "message": message,
        "requires_confirmation": requires_confirmation
    })


@app.route("/api/payment-risk/confirm", methods=["POST"])
def confirm_payment_risk():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401
    data = request.get_json(silent=True) or {}
    receiver = normalize_indian_mobile(str(data.get("receiver", "")).strip())
    try:
        amount = float(data.get("amount", 0))
    except (TypeError, ValueError):
        amount = 0
    if not receiver or not math.isfinite(amount) or amount <= 0:
        return jsonify({"success": False, "message": "Invalid payment details."}), 400

    conn = get_db()
    score, level, reasons = payment_risk_analysis(conn, session["user_id"], receiver, amount)
    if level == "HIGH":
        conn.close()
        return jsonify({"success": False, "message": "This payment is high risk and cannot be confirmed."}), 403
    if level != "MEDIUM":
        conn.close()
        return jsonify({"success": True, "message": "Risk check passed."})

    key = payment_risk_key(receiver, amount)
    session["payment_risk_approved_key"] = key
    session["payment_risk_approved_at"] = time.time()
    conn.execute("""
        INSERT INTO risk_events (user_id, receiver, amount, risk_score, risk_level, reasons, action)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (session["user_id"], receiver, amount, score, level, json.dumps(reasons), "USER_CONFIRMED"))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Risk warning confirmed. Payment can continue."})


@app.route("/api/risk-history", methods=["GET"])
def risk_history():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401
    conn = get_db()
    rows = conn.execute("""
        SELECT risk_score, risk_level, reasons, action, amount, created_at
        FROM risk_events
        WHERE user_id = ?
        ORDER BY id DESC LIMIT 20
    """, (session["user_id"],)).fetchall()
    conn.close()
    items = []
    for row in rows:
        try:
            reasons = json.loads(row["reasons"] or "[]")
        except Exception:
            reasons = []
        items.append({
            "score": row["risk_score"], "level": row["risk_level"],
            "reasons": reasons, "action": row["action"],
            "amount": row["amount"], "created_at": row["created_at"]
        })
    return jsonify({"success": True, "items": items})


# =========================================================
# FREE OTP VERIFICATION (DEMO / COLLEGE PROJECT MODE)
# =========================================================

@app.route("/api/otp/send", methods=["POST"])
def send_payment_otp():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401

    # OTP can only be requested after successful palm + PIN authorization.
    palm_verified_at = session.get("palm_verified_at")
    pin_verified_at = session.get("payment_pin_verified_at")
    now = time.time()
    if (not palm_verified_at or now - float(palm_verified_at) > 60 or
            not pin_verified_at or now - float(pin_verified_at) > 60):
        return jsonify({
            "success": False,
            "message": "Complete palm authentication and Payment PIN verification first."
        }), 403

    last_sent = session.get("payment_otp_sent_at")
    if last_sent and now - float(last_sent) < 30:
        remaining = max(1, 30 - int(now - float(last_sent)))
        return jsonify({
            "success": False,
            "message": f"Please wait {remaining} seconds before requesting another OTP.",
            "retry_after": remaining
        }), 429

    otp = f"{secrets.randbelow(1000000):06d}"
    session["payment_otp_hash"] = generate_password_hash(otp)
    session["payment_otp_sent_at"] = now
    session["payment_otp_expires_at"] = now + 120
    session["payment_otp_attempts"] = 0
    session.pop("payment_otp_verified_at", None)

    # No paid SMS service is used. The generated OTP is returned only for
    # this local/demo project so the user can test the complete OTP flow.
    log_security_activity(
        session["user_id"],
        "Payment OTP generated",
        "Free local OTP generated for payment verification"
    )

    return jsonify({
        "success": True,
        "message": "OTP generated successfully.",
        "demo_otp": otp,
        "expires_in": 120,
        "mode": "free_demo"
    })


@app.route("/api/otp/verify", methods=["POST"])
def verify_payment_otp():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401

    data = request.get_json(silent=True) or {}
    otp = str(data.get("otp", "")).strip()
    now = time.time()

    if not otp.isdigit() or len(otp) != 6:
        return jsonify({"success": False, "message": "Enter the 6-digit OTP."}), 400

    otp_hash = session.get("payment_otp_hash")
    expires_at = session.get("payment_otp_expires_at")
    attempts = int(session.get("payment_otp_attempts", 0))

    if not otp_hash or not expires_at:
        return jsonify({"success": False, "message": "Request a new OTP first."}), 400

    if now > float(expires_at):
        session.pop("payment_otp_hash", None)
        session.pop("payment_otp_expires_at", None)
        session.pop("payment_otp_attempts", None)
        return jsonify({"success": False, "message": "OTP expired. Please request a new OTP."}), 400

    if attempts >= 3:
        session.pop("payment_otp_hash", None)
        session.pop("payment_otp_expires_at", None)
        session.pop("payment_otp_attempts", None)
        return jsonify({"success": False, "message": "Too many incorrect attempts. Request a new OTP."}), 429

    if not check_password_hash(otp_hash, otp):
        session["payment_otp_attempts"] = attempts + 1
        remaining = max(0, 2 - attempts)
        return jsonify({
            "success": False,
            "message": f"Incorrect OTP. {remaining} attempts remaining." if remaining else "Incorrect OTP. Request a new OTP."
        }), 401

    session["payment_otp_verified_at"] = now
    session.pop("payment_otp_hash", None)
    session.pop("payment_otp_expires_at", None)
    session.pop("payment_otp_attempts", None)

    log_security_activity(
        session["user_id"],
        "Payment OTP verified",
        "Free OTP accepted for one payment authorization"
    )

    return jsonify({"success": True, "message": "OTP verified successfully."})


# =========================================================
# SECURITY & ACCOUNT PROTECTION
# =========================================================

def log_security_activity(user_id, activity, details=""):
    conn = get_db()
    conn.execute(
        "INSERT INTO security_activity (user_id, activity, details) VALUES (?, ?, ?)",
        (user_id, activity, details)
    )
    conn.commit()
    conn.close()


@app.route("/api/security/change-password", methods=["POST"])
def security_change_password():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401

    data = request.get_json(silent=True) or {}
    current_password = str(data.get("current_password", ""))
    new_password = str(data.get("new_password", ""))

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if not user:
        conn.close()
        return jsonify({"success": False, "message": "User not found."}), 404

    if not check_password_hash(user["password"], current_password):
        conn.close()
        log_security_activity(user["id"], "Password change failed", "Incorrect current password")
        return jsonify({"success": False, "message": "Current password is incorrect."}), 400

    if len(new_password) < 6:
        conn.close()
        return jsonify({"success": False, "message": "New password must contain at least 6 characters."}), 400

    conn.execute("UPDATE users SET password = ? WHERE id = ?", (generate_password_hash(new_password), user["id"]))
    conn.commit()
    conn.close()
    log_security_activity(user["id"], "Password changed", "Account password updated")
    return jsonify({"success": True, "message": "Password changed successfully."})


@app.route("/api/security/verify-payment-pin", methods=["POST"])
def verify_payment_pin():
    """Verify the payment PIN for one payment authorization."""
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401

    data = request.get_json(silent=True) or {}
    pin = str(data.get("pin", "")).strip()

    if not pin.isdigit() or len(pin) != 4:
        return jsonify({
            "success": False,
            "message": "Enter your 4-digit payment PIN."
        }), 400

    conn = get_db()
    user = conn.execute(
        "SELECT payment_pin FROM users WHERE id = ?",
        (session["user_id"],)
    ).fetchone()
    conn.close()

    if not user or not user["payment_pin"]:
        return jsonify({
            "success": False,
            "message": "Payment PIN is not configured. Set your PIN in Security first."
        }), 400

    if not check_password_hash(user["payment_pin"], pin):
        log_security_activity(
            session["user_id"],
            "Payment PIN verification failed",
            "Incorrect payment PIN entered during payment authorization"
        )
        return jsonify({
            "success": False,
            "message": "Incorrect payment PIN. Please try again."
        }), 401

    session["payment_pin_verified_at"] = time.time()
    log_security_activity(
        session["user_id"],
        "Payment PIN verified",
        "Payment PIN accepted for one payment authorization"
    )

    return jsonify({
        "success": True,
        "message": "Payment PIN verified."
    })


@app.route("/api/security/pin", methods=["GET", "POST"])
def security_payment_pin():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401

    if request.method == "GET":
        conn = get_db()
        user = conn.execute("SELECT payment_pin FROM users WHERE id = ?", (session["user_id"],)).fetchone()
        conn.close()
        return jsonify({"success": True, "has_pin": bool(user and user["payment_pin"])})

    data = request.get_json(silent=True) or {}
    pin = str(data.get("pin", data.get("payment_pin", ""))).strip()
    if not pin.isdigit() or len(pin) != 4:
        return jsonify({"success": False, "message": "Payment PIN must be exactly 4 digits."}), 400

    conn = get_db()
    try:
        hashed_pin = generate_password_hash(pin)
        conn.execute("UPDATE users SET payment_pin = ? WHERE id = ?", (hashed_pin, session["user_id"]))
        if conn.total_changes != 1:
            conn.rollback()
            conn.close()
            return jsonify({"success": False, "message": "Unable to save PIN for this account."}), 400
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        conn.close()
        return jsonify({"success": False, "message": "Database error while saving the PIN."}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

    log_security_activity(session["user_id"], "Payment PIN updated", "4-digit payment PIN configured")
    return jsonify({"success": True, "message": "Payment PIN saved successfully."})


@app.route("/api/security/activity")
def security_activity_api():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401

    conn = get_db()
    rows = conn.execute("""
        SELECT activity, details, created_at
        FROM security_activity
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 20
    """, (session["user_id"],)).fetchall()
    conn.close()
    return jsonify({"success": True, "activities": [dict(row) for row in rows]})


@app.route("/api/security/logout-all", methods=["POST"])
def security_logout_all():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401

    conn = get_db()
    user = conn.execute("SELECT security_version FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if not user:
        conn.close()
        return jsonify({"success": False, "message": "User not found."}), 404

    new_version = int(user["security_version"] or 1) + 1
    conn.execute("UPDATE users SET security_version = ? WHERE id = ?", (new_version, session["user_id"]))
    conn.commit()
    conn.close()
    log_security_activity(session["user_id"], "All sessions logged out", "All previous sessions were invalidated")
    session["security_version"] = new_version
    return jsonify({"success": True, "message": "All other sessions have been logged out."})


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    if "user_id" in session:
        log_security_activity(session["user_id"], "Logout", "User logged out")

    session.clear()

    return redirect(url_for("home"))


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    init_db()

    app.run(
        debug=True
    )