"""
AI Chat Company - Flask Server with Cursor API
"""
from flask import Flask, request, jsonify, session, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import os, re, secrets, requests, traceback, json, time

app = Flask(__name__, static_folder='.')

app.secret_key = os.environ.get('SECRET_KEY', 'aichat-dev-secret-2026')
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('RAILWAY_ENVIRONMENT') == 'production',
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

_db_url = os.environ.get('DATABASE_URL', 'sqlite:///aichat.db')
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)

app.config.update(
    SQLALCHEMY_DATABASE_URI=_db_url,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SQLALCHEMY_ENGINE_OPTIONS={'pool_pre_ping': True, 'pool_recycle': 300},
)

CORS(app, supports_credentials=True,
     origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','))
db = SQLAlchemy(app)

CURSOR_CHAT_URL = os.environ.get('CURSOR_PROXY_URL', 'https://api2.cursor.sh/v1/chat/completions')
NOMCHAT_URL = os.environ.get('NOMCHAT_URL', 'https://nomchat-id.up.railway.app')
OPERATOR_EMAIL = os.environ.get('OPERATOR_EMAIL', 'ai@com.ru')

CURSOR_MODELS = {
    'claude-3-7-sonnet': 'claude-3-7-sonnet-20250219',
    'claude-3-5-sonnet': 'claude-3-5-sonnet-20241022',
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    'o3-mini': 'o3-mini',
    'gemini-2.0-flash': 'gemini-2.0-flash-exp',
    'deepseek-v3': 'deepseek-chat',
    'deepseek-r1': 'deepseek-reasoner',
}

PLANS = {
    'free':  {'credits': 50,   'bonus': 300,  'label': 'Free'},
    'pro':   {'credits': 1000, 'bonus': 0,    'label': 'Pro'},
    'max':   {'credits': 5000, 'bonus': 0,    'label': 'Max'},
    'ultra': {'credits': -1,   'bonus': 0,    'label': 'Ultra'},
}

class User(db.Model):
    __tablename__ = 'users'
    id             = db.Column(db.Integer, primary_key=True)
    email          = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username       = db.Column(db.String(80), nullable=False)
    password_hash  = db.Column(db.String(256))
    nomchat_id     = db.Column(db.String(64), index=True)
    nomchat_username = db.Column(db.String(80))
    nomchat_avatar = db.Column(db.String(10), default='fox')
    credits        = db.Column(db.Integer, default=50)
    bonus_credits  = db.Column(db.Integer, default=300)
    plan           = db.Column(db.String(20), default='free')
    is_banned      = db.Column(db.Boolean, default=False)
    is_operator    = db.Column(db.Boolean, default=False)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    last_login     = db.Column(db.DateTime)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return bool(self.password_hash and check_password_hash(self.password_hash, pw))

    def to_dict(self):
        total = -1 if self.plan == 'ultra' else (self.credits or 0) + (self.bonus_credits or 0)
        return {
            'id': self.id, 'email': self.email, 'username': self.username,
            'credits': self.credits or 0, 'bonus_credits': self.bonus_credits or 0,
            'total_credits': total, 'plan': self.plan or 'free',
            'nomchat_id': self.nomchat_id, 'nomchat_username': self.nomchat_username,
            'nomchat_avatar': self.nomchat_avatar or 'fox',
            'has_password': bool(self.password_hash),
            'is_operator': self.is_operator or False,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
        }

    def deduct(self, amount=1):
        if self.plan == 'ultra': return
        if (self.bonus_credits or 0) >= amount:
            self.bonus_credits -= amount
        elif (self.credits or 0) >= amount:
            self.credits -= amount
        else:
            total = (self.bonus_credits or 0) + (self.credits or 0)
            self.bonus_credits = 0
            self.credits = max(0, total - amount)
        db.session.commit()

    def has_credits(self, amount=1):
        if self.plan == 'ultra': return True
        return ((self.credits or 0) + (self.bonus_credits or 0)) >= amount


class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    session_key = db.Column(db.String(64), unique=True, default=lambda: secrets.token_urlsafe(24))
    operator_on = db.Column(db.Boolean, default=False)
    operator_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages    = db.relationship('ChatMessage', backref='session', lazy='dynamic',
                                  cascade='all, delete-orphan')

    def to_dict(self, include_messages=False):
        d = {
            'id': self.id, 'session_key': self.session_key,
            'user_id': self.user_id, 'operator_on': self.operator_on,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_messages:
            d['messages'] = [m.to_dict() for m in self.messages.order_by(ChatMessage.created_at)]
        return d


class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    id         = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False, index=True)
    role       = db.Column(db.String(20), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id': self.id, 'role': self.role, 'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


with app.app_context():
    db.create_all()

_rate_store = {}

def rate_limit(key, max_calls=10, window=60):
    now = time.time()
    calls = [t for t in _rate_store.get(key, []) if now - t < window]
    if len(calls) >= max_calls:
        return False
    calls.append(now)
    _rate_store[key] = calls
    return True


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return jsonify({'error': 'Unauthorized'}), 401
        user = db.session.get(User, uid)
        if not user:
            session.clear()
            return jsonify({'error': 'Unauthorized'}), 401
        if user.is_banned:
            return jsonify({'error': 'banned'}), 403
        return f(*args, user=user, **kwargs)
    return decorated


def operator_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return jsonify({'error': 'Unauthorized'}), 401
        user = db.session.get(User, uid)
        if not user or (user.email != OPERATOR_EMAIL and not user.is_operator):
            return jsonify({'error': 'Forbidden'}), 403
        return f(*args, user=user, **kwargs)
    return decorated

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory('.', filename)
    except Exception:
        return jsonify({'error': 'Not found'}), 404

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f'500: {e}\n{traceback.format_exc()}')
    return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/auth/register', methods=['POST'])
def register():
    d = request.get_json(silent=True) or {}
    username = d.get('username', '').strip()
    email    = d.get('email', '').strip().lower()
    password = d.get('password', '')

    if not username or not email or not password:
        return jsonify({'error': 'All fields required'}), 400
    if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({'error': 'Invalid email'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if len(username) < 2 or len(username) > 50:
        return jsonify({'error': 'Username must be 2-50 characters'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    try:
        user = User(email=email, username=username, credits=50, bonus_credits=300)
        user.set_password(password)
        if email == OPERATOR_EMAIL:
            user.is_operator = True
        db.session.add(user)
        db.session.commit()
        session.clear()
        session['user_id'] = user.id
        session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict(), 'is_new': True})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Register error: {e}')
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    d = request.get_json(silent=True) or {}
    email    = d.get('email', '').strip().lower()
    password = d.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    try:
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid email or password'}), 401
        if user.is_banned:
            return jsonify({'error': 'Account banned'}), 403
        user.last_login = datetime.utcnow()
        db.session.commit()
        session.clear()
        session['user_id'] = user.id
        session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/nomchat', methods=['POST'])
def nomchat_auth():
    d = request.get_json(silent=True) or {}
    token = d.get('token', '').strip()
    if not token:
        return jsonify({'error': 'Token required'}), 400
    try:
        res = requests.post(
            f'{NOMCHAT_URL}/api/auth/token/verify',
            json={'token': token, 'app_id': 'ai-chat-pro'},
            timeout=8
        )
        nc = res.json()
    except Exception as e:
        return jsonify({'error': f'Nomchat unreachable: {e}'}), 503

    if not res.ok or not nc.get('success'):
        return jsonify({'error': nc.get('error', 'Invalid token')}), 401

    nc_user = nc['user']
    email   = nc_user.get('email', '').strip().lower()
    if not email:
        return jsonify({'error': 'No email from Nomchat'}), 400

    try:
        user = User.query.filter_by(email=email).first()
        is_new = user is None
        if not user:
            user = User(
                email=email,
                username=nc_user.get('username', email.split('@')[0]),
                nomchat_id=str(nc_user.get('id', '')),
                nomchat_username=nc_user.get('username', ''),
                nomchat_avatar=nc_user.get('avatar', 'fox'),
            )
            db.session.add(user)
        else:
            user.nomchat_id       = str(nc_user.get('id', ''))
            user.nomchat_username = nc_user.get('username', user.nomchat_username)
            user.nomchat_avatar   = nc_user.get('avatar', user.nomchat_avatar or 'fox')
        if user.is_banned:
            return jsonify({'error': 'Account banned'}), 403
        user.last_login = datetime.utcnow()
        db.session.commit()
        session.clear()
        session['user_id'] = user.id
        session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict(), 'is_new': is_new})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Authentication failed'}), 500


@app.route('/api/auth/me')
@login_required
def me(user):
    return jsonify(user.to_dict())


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

def get_cursor_cookie():
    cookies = os.environ.get('CURSOR_COOKIE', '')
    if not cookies:
        return None
    cookie_list = [c.strip() for c in cookies.split(',') if c.strip()]
    if not cookie_list:
        return None
    idx = int(time.time()) % len(cookie_list)
    return cookie_list[idx]


def call_cursor_api(messages, model='claude-3-7-sonnet', stream=False):
    cookie = get_cursor_cookie()
    if not cookie:
        raise ValueError('CURSOR_COOKIE not configured')
    cursor_model = CURSOR_MODELS.get(model, model)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {cookie}',
    }
    payload = {
        'model': cursor_model,
        'messages': messages,
        'stream': stream,
        'temperature': 0.7,
        'max_tokens': 4096,
    }
    resp = requests.post(CURSOR_CHAT_URL, headers=headers, json=payload, stream=stream, timeout=120)
    return resp


@app.route('/api/config/models')
def get_models():
    return jsonify({'models': list(CURSOR_MODELS.keys()), 'default': 'claude-3-7-sonnet'})


@app.route('/api/chat', methods=['POST'])
@login_required
def chat(user):
    if not user.has_credits():
        return jsonify({'error': 'no_credits', 'message': 'No credits remaining.'}), 402

    cookie = get_cursor_cookie()
    if not cookie:
        return jsonify({'error': 'AI service not configured. Set CURSOR_COOKIE env variable.'}), 503

    d = request.get_json(silent=True) or {}
    messages = d.get('messages', [])
    model    = d.get('model', 'claude-3-7-sonnet')
    do_stream = d.get('stream', False)

    if model not in CURSOR_MODELS:
        model = 'claude-3-7-sonnet'

    clean_msgs = []
    for m in messages:
        role    = m.get('role', 'user')
        content = m.get('content', '')
        if role not in ('system', 'user', 'assistant'):
            continue
        if isinstance(content, str) and content.strip():
            clean_msgs.append({'role': role, 'content': content.strip()})
        elif isinstance(content, list):
            clean_msgs.append({'role': role, 'content': content})

    if not clean_msgs:
        return jsonify({'error': 'No messages provided'}), 400

    try:
        if do_stream:
            def generate():
                try:
                    resp = call_cursor_api(clean_msgs, model, stream=True)
                    if not resp.ok:
                        err_text = resp.text[:200]
                        yield f"data: {json.dumps({'error': f'Cursor API error {resp.status_code}: {err_text}'})}\n\n"
                        return
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode('utf-8') if isinstance(line, bytes) else line
                            yield decoded + '\n\n'
                    user.deduct()
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return Response(
                stream_with_context(generate()),
                content_type='text/event-stream',
                headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'}
            )
        else:
            resp = call_cursor_api(clean_msgs, model, stream=False)
            if not resp.ok:
                err_msg = f'Cursor API error {resp.status_code}'
                try:
                    err_data = resp.json()
                    if isinstance(err_data.get('error'), dict):
                        err_msg = err_data['error'].get('message', err_msg)
                    else:
                        err_msg = err_data.get('error', err_msg)
                except Exception:
                    pass
                return jsonify({'error': err_msg}), resp.status_code
            result = resp.json()
            user.deduct()
            total = -1 if user.plan == 'ultra' else (user.credits or 0) + (user.bonus_credits or 0)
            result['credits_remaining'] = total
            return jsonify(result)
    except requests.Timeout:
        return jsonify({'error': 'AI request timed out. Please try again.'}), 504
    except Exception as e:
        app.logger.error(f'Chat error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/user/update', methods=['POST'])
@login_required
def update_user(user):
    d = request.get_json(silent=True) or {}
    if 'username' in d:
        name = d['username'].strip()
        if 2 <= len(name) <= 50:
            user.username = name
    if 'password' in d and len(d['password']) >= 6:
        user.set_password(d['password'])
    db.session.commit()
    return jsonify(user.to_dict())


@app.route('/api/user/delete', methods=['DELETE'])
@login_required
def delete_user(user):
    db.session.delete(user)
    db.session.commit()
    session.clear()
    return jsonify({'success': True})


@app.route('/api/plans')
def get_plans():
    return jsonify(PLANS)
